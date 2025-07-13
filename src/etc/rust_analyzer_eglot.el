((rustic-mode
  .((eglot-workspace-configuration
     . (:rust-analyzer
        ( :check ( :invocationLocation "root"
                                       :invocationStrategy "once"
                                       :overrideCommand ["python3"
                                                         "x.py"
                                                         "check"
                                                         "--json-output"])
                 :linkedProjects ["Cargo.toml"
                                  "compiler/rustc_codegen_cranelift/Cargo.toml"
                                  "compiler/rustc_codegen_gcc/Cargo.toml"
                                  "library/Cargo.toml"
                                  "src/bootstrap/Cargo.toml"
                                  "src/tools/rust-analyzer/Cargo.toml"]
                 :rustfmt ( :overrideCommand ["build/host/rustfmt/bin/rustfmt"
                                              "--edition=2024"])
                 :procMacro ( :server "build/host/stage0/libexec/rust-analyzer-proc-macro-srv"
                                      :enable t)
                 :cargo ( :buildScripts ( :enable t
                                                  :invocationLocation "root"
                                                  :invocationStrategy "once"
                                                  :overrideCommand ["python3"
                                                                    "x.py"
                                                                    "check"
                                                                    "--json-output"
                                                                    "--compile-time-deps"])]
                                        :sysrootSrc "./library"
                                        :extraEnv (:RUSTC_BOOTSTRAP "1"))
                 :rustc ( :source "./Cargo.toml" )))))))
