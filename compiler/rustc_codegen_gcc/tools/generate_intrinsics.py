import json
import os
import re
import sys
import subprocess


def run_command(command, cwd=None):
    p = subprocess.Popen(command, cwd=cwd)
    if p.wait() != 0:
        print("command `{}` failed...".format(" ".join(command)))
        sys.exit(1)


def clone_repository(repo_name, path, repo_url, branch="master", sub_paths=None):
    if os.path.exists(path):
        while True:
            choice = input("There is already a `{}` folder, do you want to update it? [y/N]".format(path))
            if choice == "" or choice.lower() == "n":
                print("Skipping repository update.")
                return
            elif choice.lower() == "y":
                print("Updating repository...")
                run_command(["git", "pull", "origin", branch], cwd=path)
                return
            else:
                print("Didn't understand answer...")
    print("Cloning {} repository...".format(repo_name))
    if sub_paths is None:
        run_command(["git", "clone", repo_url, "--depth", "1", path])
    else:
        run_command(["git", "clone", repo_url, "--filter=tree:0", "--no-checkout", path])
        run_command(["git", "sparse-checkout", "init"], cwd=path)
        run_command(["git", "sparse-checkout", "set", *sub_paths], cwd=path)
        run_command(["git", "checkout"], cwd=path)


def append_intrinsic(array, intrinsic_name, translation):
    array.append((intrinsic_name, translation))


def convert_to_string(content):
    if content.__class__.__name__ == 'bytes':
        return content.decode('utf-8')
    return content


def extract_intrinsics_from_llvm(llvm_path, intrinsics):
    command = ["llvm-tblgen", "llvm/IR/Intrinsics.td"]
    cwd = os.path.join(llvm_path, "llvm/include")
    print("=> Running command `{}` from `{}`".format(command, cwd))
    p = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    lines = convert_to_string(output).splitlines()
    pos = 0
    while pos < len(lines):
        line = lines[pos]
        if not line.startswith("def "):
            pos += 1
            continue
        intrinsic = line.split(" ")[1].strip()
        content = line
        while pos < len(lines):
            line = lines[pos].split(" // ")[0].strip()
            content += line
            pos += 1
            if line == "}":
                break
        entries = re.findall('string ClangBuiltinName = "(\\w+)";', content)
        current_arch = re.findall('string TargetPrefix = "(\\w+)";', content)
        if len(entries) == 1 and len(current_arch) == 1:
            current_arch = current_arch[0]
            intrinsic = intrinsic.split("_")
            if len(intrinsic) < 2 or intrinsic[0] != "int":
                continue
            intrinsic[0] = "llvm"
            intrinsic = ".".join(intrinsic)
            if current_arch not in intrinsics:
                intrinsics[current_arch] = []
            append_intrinsic(intrinsics[current_arch], intrinsic, entries[0])


def append_translation(json_data, p, array):
    it = json_data["index"][p]
    content = it["docs"].split('`')
    if len(content) != 5:
        return
    append_intrinsic(array, content[1], content[3])


def extract_intrinsics_from_llvmint(llvmint, intrinsics):
    archs = [
        "AMDGPU",
        "aarch64",
        "arm",
        "cuda",
        "hexagon",
        "mips",
        "nvvm",
        "ppc",
        "ptx",
        "x86",
        "xcore",
    ]

    json_file = os.path.join(llvmint, "target/doc/llvmint.json")
    # We need to regenerate the documentation!
    run_command(
        ["cargo", "rustdoc", "--", "-Zunstable-options", "--output-format", "json"],
        cwd=llvmint,
    )
    with open(json_file, "r", encoding="utf8") as f:
        json_data = json.loads(f.read())
    for p in json_data["paths"]:
        it = json_data["paths"][p]
        if it["crate_id"] != 0:
            # This is from an external crate.
            continue
        if it["kind"] != "function":
            # We're only looking for functions.
            continue
        # if len(it["path"]) == 2:
        #   # This is a "general" intrinsic, not bound to a specific arch.
        #   append_translation(json_data, p, general)
        #   continue
        if len(it["path"]) != 3 or it["path"][1] not in archs:
            continue
        arch = it["path"][1]
        if arch not in intrinsics:
            intrinsics[arch] = []
        append_translation(json_data, p, intrinsics[arch])


def fill_intrinsics(intrinsics, from_intrinsics, all_intrinsics):
    for arch in from_intrinsics:
        if arch not in intrinsics:
            intrinsics[arch] = []
        for entry in from_intrinsics[arch]:
            if entry[0] in all_intrinsics:
                if all_intrinsics[entry[0]] == entry[1]:
                    # This is a "full" duplicate, both the LLVM instruction and the GCC
                    # translation are the same.
                    continue
                intrinsics[arch].append((entry[0], entry[1], True))
            else:
                intrinsics[arch].append((entry[0], entry[1], False))
                all_intrinsics[entry[0]] = entry[1]


def update_intrinsics(llvm_path, llvmint, llvmint2):
    intrinsics_llvm = {}
    intrinsics_llvmint = {}
    all_intrinsics = {}

    extract_intrinsics_from_llvm(llvm_path, intrinsics_llvm)
    extract_intrinsics_from_llvmint(llvmint, intrinsics_llvmint)
    extract_intrinsics_from_llvmint(llvmint2, intrinsics_llvmint)

    intrinsics = {}
    # We give priority to translations from LLVM over the ones from llvmint.
    fill_intrinsics(intrinsics, intrinsics_llvm, all_intrinsics)
    fill_intrinsics(intrinsics, intrinsics_llvmint, all_intrinsics)

    archs = [arch for arch in intrinsics]
    archs.sort()

    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../src/intrinsic/archs.rs",
    )
    # A hashmap of all architectures. This allows us to first match on the architecture, and then on the intrinsics.
    # This speeds up the comparison, and makes our code considerably smaller.
    # Since all intrinsic names start with "llvm.", we skip that prefix.
    print("Updating content of `{}`...".format(output_file))
    with open(output_file, "w", encoding="utf8") as out:
        out.write("// File generated by `rustc_codegen_gcc/tools/generate_intrinsics.py`\n")
        out.write("// DO NOT EDIT IT!\n")
        out.write("/// Translate a given LLVM intrinsic name to an equivalent GCC one.\n")
        out.write("fn map_arch_intrinsic(full_name:&str)->&'static str{\n")
        out.write('let Some(name) = full_name.strip_prefix("llvm.") else { unimplemented!("***** unsupported LLVM intrinsic {}", full_name) };\n')
        out.write('let Some((arch, name)) = name.split_once(\'.\') else { unimplemented!("***** unsupported LLVM intrinsic {}", name) };\n')
        out.write("match arch {\n")
        for arch in archs:
            if len(intrinsics[arch]) == 0:
                continue
            out.write("\"{}\" => {{ #[allow(non_snake_case)] fn {}(name: &str,full_name:&str) -> &'static str {{ match name {{".format(arch,arch))
            intrinsics[arch].sort(key=lambda x: (x[0], x[2]))
            out.write('    // {}\n'.format(arch))
            for entry in intrinsics[arch]:
                llvm_name = entry[0].removeprefix("llvm.");
                llvm_name = llvm_name.removeprefix(arch);
                llvm_name = llvm_name.removeprefix(".");
                if entry[2] is True: # if it is a duplicate
                    out.write('    // [DUPLICATE]: "{}" => "{}",\n'.format(llvm_name, entry[1]))
                elif "_round_mask" in entry[1]:
                    out.write('    // [INVALID CONVERSION]: "{}" => "{}",\n'.format(llvm_name, entry[1]))
                else:
                    out.write('    "{}" => "{}",\n'.format(llvm_name, entry[1]))
            out.write('    _ => unimplemented!("***** unsupported LLVM intrinsic {full_name}"),\n')
            out.write("}} }} {}(name,full_name) }}\n,".format(arch))
        out.write('    _ => unimplemented!("***** unsupported LLVM architecture {arch}, intrinsic:{full_name}"),\n')
        out.write("}\n}")
    subprocess.call(["rustfmt", output_file])
    print("Done!")


def main():
    llvm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "llvm-project",
    )
    llvmint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "llvmint",
    )
    llvmint2_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "llvmint-2",
    )

    # First, we clone the LLVM repository if it's not already here.
    clone_repository(
        "llvm-project",
        llvm_path,
        "https://github.com/llvm/llvm-project",
        branch="main",
        sub_paths=["llvm/include/llvm/IR", "llvm/include/llvm/CodeGen/"],
    )
    clone_repository(
        "llvmint",
        llvmint_path,
        "https://github.com/GuillaumeGomez/llvmint",
    )
    clone_repository(
        "llvmint2",
        llvmint2_path,
        "https://github.com/antoyo/llvmint",
    )
    update_intrinsics(llvm_path, llvmint_path, llvmint2_path)


if __name__ == "__main__":
    sys.exit(main())
