/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#![feature(rustc_private)]

use std::env;
use std::path::{Path, PathBuf};

use rustc_driver::{Callbacks, run_compiler};
use rustc_interface::interface;
use rustc_session::config;
use rustc_target::spec::Target as RustcTarget;
use tracing::{debug, info};

/// Custom callbacks that register the MLIR codegen backend programmatically
struct MlirBackendCallbacks;

impl Callbacks for MlirBackendCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        debug!("MlirBackendCallbacks::config called - registering backend");
        // Register the MLIR codegen backend programmatically
        // This closure will be called when rustc needs to create the codegen backend
        config.make_codegen_backend =
            Some(Box::new(|_opts: &config::Options, _target: &RustcTarget| {
                debug!("make_codegen_backend closure called - creating MlirCodegenBackend");
                // Create and return the MLIR codegen backend
                rustc_codegen_llvm::mlir::MlirCodegenBackend::new()
            }));
    }
}

#[derive(Debug, Clone, Default)]
pub struct LlvmCompiler {}

impl LlvmCompiler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compile(&self, filename: &Path, target: &str) -> Result<(), Box<dyn std::error::Error>> {
        let working_dir = PathBuf::from("/tmp");

        // Use custom callbacks that register the MLIR backend
        let mut callbacks = MlirBackendCallbacks;
        let exe_name = "/home/arshadm/.cargo/bin/rustc".to_string(); // AXM FIXME: remove this once API changes
        let output = format!("-o{}", working_dir.join("kernel.asm").display());
        let build_type = "-Copt-level=3".to_string(); // Use opt-level=3 for release build
        let panic_abort = "-Cpanic=abort".to_string();
        let target = format!("--target={}", target);
        let crate_type = "--crate-type=lib".to_string();
        // let emit = "--emit=llvm-ir".to_string();
        let overflow_checks = "-C".to_string();
        let overflow_checks_off = "overflow-checks=off".to_string();
        let frontend = "--frontend=triton".to_string();

        info!("Working directory: {}", working_dir.display());
        info!("Target: {}", target);
        info!("Output: {}", output);
        debug!(
            "Rustc command: {} {} {} {} {}",
            exe_name,
            filename.display(),
            output,
            target,
            crate_type
        );

        unsafe {
            env::set_var("CFG_VERSION", "tg-1.90.0");
        }

        // Build the arguments for the compiler
        // Note: We no longer need -Zcodegen-backend flag since we're registering
        // the backend programmatically via the callbacks
        let args = vec![
            exe_name,
            filename.display().to_string(),
            build_type,
            panic_abort,
            output,
            target,
            crate_type,
            // emit,
            overflow_checks,
            overflow_checks_off,
            frontend,
        ];

        run_compiler(&args, &mut callbacks);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::{EnvFilter, fmt};

    use super::*;

    #[test]
    fn test_triton_tensor_add() {
        // Set RUST_LOG=debug in environment to see debug output
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let tensor_add = env::current_dir().unwrap().join("tests/data/triton_tensor_add.rs");
        let target = "nvptx64-nvidia-cuda";
        println!("Compiling tensor add with target: {}", tensor_add.display());
        let result = compiler.compile(&tensor_add, target);
        assert!(result.is_ok());
    }
}
