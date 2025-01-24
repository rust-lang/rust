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
                                  "src/tools/x/Cargo.toml"
                                  "src/bootstrap/Cargo.toml"
                                  "src/tools/rust-analyzer/Cargo.toml"
                                  "compiler/rustc_codegen_cranelift/Cargo.toml"
                                  "compiler/rustc_codegen_gcc/Cargo.toml"]
                 :rustfmt ( :overrideCommand ["build/host/rustfmt/bin/rustfmt"
                                              "--edition=2021"])
                 :procMacro ( :server "build/host/stage0/libexec/rust-analyzer-proc-macro-srv"
                                      :enable t)
                 :cargo ( :buildScripts ( :enable t
                                                  :invocationLocation "root"
                                                  :invocationStrategy "once"
                                                  :overrideCommand ["python3"
                                                                    "x.py"
                                                                    "check"
                                                                    "--json-output"])
                                        :sysrootSrc "./library"
                                        :extraEnv (:RUSTC_BOOTSTRAP "1"))
                 :rustc ( :source "./Cargo.toml" )))))))
