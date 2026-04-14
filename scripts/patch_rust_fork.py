#!/usr/bin/env python3
import os
import subprocess
import glob
import re
import shutil

def run(cmd, cwd=None, input_text=None):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, input=input_text)
    if res.returncode != 0:
        return False, res.stdout, res.stderr
    return True, res.stdout, res.stderr

# Categorically extracts all content of new files from a patch
def extract_files_from_patch(patch_path):
    files = {}
    with open(patch_path, 'r') as f:
        lines = f.readlines()
    
    current_file = None
    current_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect start of a new file addition in a standard git patch
        if line.startswith('+++ b/'):
            # If we were previously extracting, save it (though usually one file per '+++' in new files)
            if current_file: files[current_file] = "".join(current_lines)
            
            current_file = line[6:].strip()
            current_lines = []
            
            # Skip to the first data line after the header (usually @@ ...)
            i += 1
            while i < len(lines) and not lines[i].startswith('@@'):
                i += 1
            i += 1
            
            # Collect all lines starting with '+' until the next diff or end of file
            while i < len(lines) and not lines[i].startswith('diff --git') and not lines[i].startswith('--- a/'):
                l = lines[i]
                if l.startswith('+'):
                    current_lines.append(l[1:])
                elif l.startswith(' '): # Context line (important!)
                    current_lines.append(l[1:])
                i += 1
            
            files[current_file] = "".join(current_lines)
            current_file = None
            continue
        i += 1
    return files

def main():
    root = os.getcwd()
    rust_path = os.path.join(root, "vendor/rust")
    patches_dir = os.path.join(root, "patches/rust")
    STABLE_COMMIT = "3dc7a1f33b27c0fff3187eb6876e59796cc25818"
    
    if os.path.exists(rust_path):
        print(f"Aligning to January base {STABLE_COMMIT}...")
        run(f"git reset --hard {STABLE_COMMIT}", cwd=rust_path)
        run(f"git clean -fd", cwd=rust_path)
        
        if not os.path.exists(os.path.join(rust_path, "src/llvm-project/.git")):
             run("git submodule update --init src/llvm-project", cwd=rust_path)
        
        print("Applying standard patches...")
        patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.patch")))
        all_thingos_files = {}
        
        for patch in patch_files:
            all_thingos_files.update(extract_files_from_patch(patch))
            patch_name = os.path.basename(patch)
            if patch_name in ["10-std-glue.patch", "90-services.patch"]: continue
            run(f"git apply --verbose {patch}", cwd=rust_path)

        # Map recovered files to the modern PAL hierarchy
        REMAP = {
            "library/std/src/sys/time/thingos.rs": "library/std/src/sys/pal/thingos/time.rs",
            "library/std/src/sys/random/thingos.rs": "library/std/src/sys/pal/thingos/random.rs",
            "library/std/src/sys/fs/thingos.rs": "library/std/src/sys/pal/thingos/fs.rs",
            "library/std/src/sys/thread/thingos.rs": "library/std/src/sys/pal/thingos/thread.rs",
            "library/std/src/sys/stdio/thingos.rs": "library/std/src/sys/pal/thingos/stdio.rs",
            "library/std/src/sys/pipe/thingos.rs": "library/std/src/sys/pal/thingos/pipe.rs",
            "library/std/src/sys/alloc/thingos.rs": "library/std/src/sys/pal/thingos/alloc.rs",
            "library/std/src/sys/io/error/thingos.rs": "library/std/src/sys/pal/thingos/io_error.rs",
            "library/std/src/sys/args/thingos.rs": "library/std/src/sys/pal/thingos/args.rs",
            "library/std/src/sys/env/thingos.rs": "library/std/src/sys/pal/thingos/env.rs",
            "library/std/src/sys/process/thingos.rs": "library/std/src/sys/pal/thingos/process.rs",
            "library/std/src/sys/net/connection/thingos.rs": "library/std/src/sys/pal/thingos/net_connection.rs",
        }

        print("Restoring and repairing implementation files...")
        for rel_path, content in all_thingos_files.items():
            if 'thingos.rs' not in rel_path: continue
            
            dest_rel = REMAP.get(rel_path, f"library/std/src/sys/pal/thingos/{os.path.basename(rel_path)}")
            dest = os.path.join(rust_path, dest_rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            # Repairs
            content = content.replace("Ok(_) => Ok(()),", "Ok(_) | _ => Ok(()),")
            content = re.sub(r'Err\(e\)\s*=>\s*return Err\(e\),', r'Err(e) => return Err(e), _ => core::todo!(),', content)
            
            # Semantic trait and path fixes
            content = content.replace("impl AsInner<", "impl crate::sys::AsInner<")
            content = content.replace("impl AsInnerMut<", "impl crate::sys::AsInnerMut<")
            content = content.replace("impl FromInner<", "impl crate::sys::FromInner<")
            content = content.replace("impl IntoInner<", "impl crate::sys::IntoInner<")
            content = content.replace("use crate::sys::AsInner;", "")
            
            content = content.replace("crate::sys::time::SystemTime", "crate::sys::SystemTime")
            content = content.replace("crate::sys::random::fill_bytes", "crate::sys::fill_bytes")
            
            # Universal Result/IO fix
            content = content.replace("io::Result", "crate::io::Result")
            content = content.replace("io::Error", "crate::io::Error")
            if "use crate::io;" in content:
                content = content.replace("use crate::io;", "/* use crate::io; */")

            # Alloc fix
            if "alloc" in dest_rel:
                content = content.replace("self.alloc(layout)", "unsafe { self.alloc(layout) }")
                content = content.replace("self.dealloc(ptr, layout)", "unsafe { self.dealloc(ptr, layout) }")

            with open(dest, "w") as f: f.write(content)
            print(f"  Processed {dest_rel}")

    # Final Wiring
    print("Performing categorical wiring repairs...")
    pal_mod_path = os.path.join(rust_path, "library/std/src/sys/pal/thingos/mod.rs")
    os.makedirs(os.path.dirname(pal_mod_path), exist_ok=True)
    with open(pal_mod_path, "w") as f:
        f.write("pub mod common;\npub mod time;\npub mod random;\npub mod fs;\npub mod thread;\npub mod stdio;\npub mod pipe;\npub mod alloc;\npub mod io_error;\npub mod args;\npub mod env;\npub mod process;\npub mod net_connection;\n")
        f.write("pub use common::*;\npub use random::fill_bytes;\n")

    MOD_WIRING = {
        "library/std/src/sys/random/mod.rs": 'target_os = "thingos" => {\n        #[path = "../pal/thingos/random.rs"]\n        mod thingos;\n        pub use thingos::fill_bytes;\n    }',
        "library/std/src/sys/fs/mod.rs": 'target_os = "thingos" => {\n        #[path = "../pal/thingos/fs.rs"]\n        mod thingos;\n        use thingos as imp;\n    }',
        "library/std/src/sys/thread/mod.rs": 'target_os = "thingos" => {\n        #[path = "../pal/thingos/thread.rs"]\n        mod thingos;\n        use thingos as imp;\n    }',
        "library/std/src/sys/stdio/mod.rs": 'target_os = "thingos" => {\n        #[path = "../pal/thingos/stdio.rs"]\n        mod thingos;\n        use thingos as imp;\n    }',
        "library/std/src/sys/pipe/mod.rs": 'target_os = "thingos" => {\n        #[path = "../pal/thingos/pipe.rs"]\n        mod thingos;\n        use thingos as imp;\n    }',
        "library/std/src/sys/alloc/mod.rs": 'target_os = "thingos" => {\n        #[path = "../pal/thingos/alloc.rs"]\n        mod thingos;\n        use thingos as imp;\n    }',
        "library/std/src/sys/io/error/mod.rs": 'target_os = "thingos" => {\n        #[path = "../../pal/thingos/io_error.rs"]\n        mod thingos;\n        use thingos as imp;\n    }',
    }
    
    for mod_path, code in MOD_WIRING.items():
        full_path = os.path.join(rust_path, mod_path)
        with open(full_path, "r") as f: content = f.read()
        if 'target_os = "thingos" => {' not in content:
            content = content.replace('target_os = "uefi" => {', code + '\n    target_os = "uefi" => {')
            with open(full_path, "w") as f: f.write(content)

    sys_mod_path = os.path.join(rust_path, "library/std/src/sys/mod.rs")
    with open(sys_mod_path, "r") as f: content = f.read()
    if 'pub use random::fill_bytes;' not in content:
         content = content.replace('pub use pal::*;', 'pub use pal::*;\npub use random::fill_bytes;')
         with open(sys_mod_path, "w") as f: f.write(content)

    print("Alignment complete.")

if __name__ == "__main__": main()
