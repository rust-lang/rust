This tool extracts coverage mapping information from an LLVM IR assembly file
(`.ll`), and prints it in a more human-readable form that can be used for
snapshot tests.

The output format is mostly arbitrary, so it's OK to change the output as long
as any affected tests are also re-blessed. However, the output should be
consistent across different executions on different platforms, so avoid
printing any information that is platform-specific or non-deterministic.

## Demangle mode

When run as `coverage-dump --demangle`, this tool instead functions as a
command-line demangler that can be invoked by `llvm-cov`.
