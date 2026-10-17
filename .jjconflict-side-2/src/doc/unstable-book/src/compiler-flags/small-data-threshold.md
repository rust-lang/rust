# `small-data-threshold`

-----------------------

This flag controls the maximum static variable size that may be included in the
"small data sections" (.sdata, .sbss) supported by some architectures (RISCV,
MIPS, M68K, Hexagon).  Can be set to `0` to disable the use of small data
sections.

Target support is indicated by the `small_data_threshold_support` target
option which can be:

- `none` (`SmallDataThresholdSupport::None`) for no support
- `default-for-arch` (`SmallDataThresholdSupport::DefaultForArch`) which
  is automatically translated into an appropriate value for the target.
- `llvm-module-flag=<flag_name>`
  (`SmallDataThresholdSupport::LlvmModuleFlag`) for specifying the
  threshold via an LLVM module flag
- `llvm-arg=<arg_name>` (`SmallDataThresholdSupport::LlvmArg`) for
  specifying the threshold via an LLVM argument.
