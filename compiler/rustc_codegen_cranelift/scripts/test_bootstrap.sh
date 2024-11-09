#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../"

source ./scripts/setup_rust_fork.sh

echo "[TEST] Bootstrap of rustc"
pushd rust
rm -r compiler/rustc_codegen_cranelift/{Cargo.*,src}
cp ../Cargo.* compiler/rustc_codegen_cranelift/
cp -r ../src compiler/rustc_codegen_cranelift/src

# FIXME(rust-lang/rust#132719) remove once it doesn't break without this patch
cat <<EOF | git apply -
diff --git a/src/bootstrap/src/core/build_steps/compile.rs b/src/bootstrap/src/core/build_steps/compile.rs
index 3394f2a84a0..cb980dd4d7c 100644
--- a/src/bootstrap/src/core/build_steps/compile.rs
+++ b/src/bootstrap/src/core/build_steps/compile.rs
@@ -1976,7 +1976,7 @@ fn run(self, builder: &Builder<'_>) -> Compiler {
             }
         }

-        {
+        if builder.config.llvm_enabled(target_compiler.host) && builder.config.llvm_tools_enabled {
             // \`llvm-strip\` is used by rustc, which is actually just a symlink to \`llvm-objcopy\`,
             // so copy and rename \`llvm-objcopy\`.
             let src_exe = exe("llvm-objcopy", target_compiler.host);
EOF

./x.py build --stage 1 library/std
popd
