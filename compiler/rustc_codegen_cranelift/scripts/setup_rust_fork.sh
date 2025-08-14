#!/usr/bin/env bash
set -e

# CG_CLIF_FORCE_GNU_AS will force usage of as instead of the LLVM backend of rustc as
# the LLVM backend isn't compiled in here.
export CG_CLIF_FORCE_GNU_AS=1

# Compiletest expects all standard library paths to start with /rustc/FAKE_PREFIX.
# CG_CLIF_STDLIB_REMAP_PATH_PREFIX will cause cg_clif's build system to pass
# --remap-path-prefix to handle this.
CG_CLIF_STDLIB_REMAP_PATH_PREFIX=/rustc/FAKE_PREFIX ./y.sh build

echo "[SETUP] Rust fork"
git clone --quiet https://github.com/rust-lang/rust.git --filter=tree:0 || true
pushd rust
git fetch
git checkout --no-progress -- .
git checkout --no-progress "$(rustc -V | cut -d' ' -f3 | tr -d '(')"

git submodule update --quiet --init src/tools/cargo library/backtrace library/stdarch

git -c user.name=Dummy -c user.email=dummy@example.com -c commit.gpgSign=false \
    am ../patches/*-stdlib-*.patch

cat > config.toml <<EOF
change-id = 999999

[llvm]
download-ci-llvm = true

[build]
rustc = "$(pwd)/../dist/bin/rustc-clif"
cargo = "$(rustup which cargo)"
full-bootstrap = true
local-rebuild = true
compiletest-allow-stage0 = true

[rust]
download-rustc = false
codegen-backends = ["cranelift"]
deny-warnings = false
verbose-tests = false
# The cg_clif sysroot doesn't contain llvm tools and unless llvm_tools is
# disabled bootstrap will crash trying to copy llvm tools for the bootstrap
# compiler.
llvm-tools = false
std-features = ["panic-unwind"]

EOF

cat <<EOF | git apply -
diff --git a/src/bootstrap/src/core/config/config.rs b/src/bootstrap/src/core/config/config.rs
index cf4ef4ee310..fe78560fcaf 100644
--- a/src/bootstrap/src/core/config/config.rs
+++ b/src/bootstrap/src/core/config/config.rs
@@ -3138,13 +3138,6 @@ fn parse_download_ci_llvm(
                     );
                 }

-                if b && self.is_running_on_ci {
-                    // On CI, we must always rebuild LLVM if there were any modifications to it
-                    panic!(
-                        "\`llvm.download-ci-llvm\` cannot be set to \`true\` on CI. Use \`if-unchanged\` instead."
-                    );
-                }
-
                 // If download-ci-llvm=true we also want to check that CI llvm is available
                 b && llvm::is_ci_llvm_available_for_target(self, asserts)
             }
EOF

popd

# Allow the testsuite to use llvm tools
host_triple=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")
export LLVM_BIN_DIR="$(rustc --print sysroot)/lib/rustlib/$host_triple/bin"
