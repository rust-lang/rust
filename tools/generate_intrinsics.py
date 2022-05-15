import json
import os
import re
import sys
import subprocess
from os import walk


def run_command(command, cwd=None):
    p = subprocess.Popen(command, cwd=cwd)
    if p.wait() != 0:
        print("command `{}` failed...".format(" ".join(command)))
        sys.exit(1)


def clone_repository(repo_name, path, repo_url, sub_path=None):
    if os.path.exists(path):
        while True:
            choice = input("There is already a `{}` folder, do you want to update it? [y/N]".format(path))
            if choice == "" or choice.lower() == "n":
                print("Skipping repository update.")
                return
            elif choice.lower() == "y":
                print("Updating repository...")
                run_command(["git", "pull", "origin"], cwd=path)
                return
            else:
                print("Didn't understand answer...")
    print("Cloning {} repository...".format(repo_name))
    if sub_path is None:
        run_command(["git", "clone", repo_url, "--depth", "1", path])
    else:
        run_command(["git", "clone", repo_url, "--filter=tree:0", "--no-checkout", path])
        run_command(["git", "sparse-checkout", "init"], cwd=path)
        run_command(["git", "sparse-checkout", "set", "add", sub_path], cwd=path)
        run_command(["git", "checkout"], cwd=path)


def append_intrinsic(array, intrinsic_name, translation):
    array.append((intrinsic_name, translation))


def extract_instrinsics(intrinsics, file):
    print("Extracting intrinsics from `{}`...".format(file))
    with open(file, "r", encoding="utf8") as f:
        content = f.read()

    lines = content.splitlines()
    pos = 0
    current_arch = None
    while pos < len(lines):
        line = lines[pos].strip()
        if line.startswith("let TargetPrefix ="):
            current_arch = line.split('"')[1].strip()
            if len(current_arch) == 0:
                current_arch = None
        elif current_arch is None:
            pass
        elif line == "}":
            current_arch = None
        elif line.startswith("def "):
            content = ""
            while not content.endswith(";") and not content.endswith("}") and pos < len(lines):
                line = lines[pos].split(" // ")[0].strip()
                content += line
                pos += 1
            entries = re.findall('GCCBuiltin<"(\\w+)">', content)
            if len(entries) > 0:
                intrinsic = content.split("def ")[1].strip().split(":")[0].strip()
                intrinsic = intrinsic.split("_")
                if len(intrinsic) < 2 or intrinsic[0] != "int":
                    continue
                intrinsic[0] = "llvm"
                intrinsic = ".".join(intrinsic)
                if current_arch not in intrinsics:
                    intrinsics[current_arch] = []
                for entry in entries:
                    append_intrinsic(intrinsics[current_arch], intrinsic, entry)
            continue
        pos += 1
        continue
    print("Done!")


def extract_instrinsics_from_llvm(llvm_path, intrinsics):
    files = []
    intrinsics_path = os.path.join(llvm_path, "llvm/include/llvm/IR")
    for (dirpath, dirnames, filenames) in walk(intrinsics_path):
        files.extend([os.path.join(intrinsics_path, f) for f in filenames if f.endswith(".td")])

    for file in files:
        extract_instrinsics(intrinsics, file)


def append_translation(json_data, p, array):
    it = json_data["index"][p]
    content = it["docs"].split('`')
    if len(content) != 5:
        return
    append_intrinsic(array, content[1], content[3])


def extract_instrinsics_from_llvmint(llvmint, intrinsics):
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

    extract_instrinsics_from_llvm(llvm_path, intrinsics_llvm)
    extract_instrinsics_from_llvmint(llvmint, intrinsics_llvmint)
    extract_instrinsics_from_llvmint(llvmint2, intrinsics_llvmint)

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
    print("Updating content of `{}`...".format(output_file))
    with open(output_file, "w", encoding="utf8") as out:
        out.write("// File generated by `rustc_codegen_gcc/tools/generate_intrinsics.py`\n")
        out.write("// DO NOT EDIT IT!\n")
        out.write("match name {\n")
        for arch in archs:
            if len(intrinsics[arch]) == 0:
                continue
            intrinsics[arch].sort(key=lambda x: (x[0], x[2]))
            out.write('    // {}\n'.format(arch))
            for entry in intrinsics[arch]:
                if entry[2] == True: # if it is a duplicate
                    out.write('    // [DUPLICATE]: "{}" => "{}",\n'.format(entry[0], entry[1]))
                else:
                    out.write('    "{}" => "{}",\n'.format(entry[0], entry[1]))
        out.write('    _ => unimplemented!("***** unsupported LLVM intrinsic {}", name),\n')
        out.write("}\n")
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
        sub_path="llvm/include/llvm/IR",
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
