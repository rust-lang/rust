# Additional Resources

## ABIs and Platform Specifics

### System V

The System V ABIs are defined by a General ABI (gABI),
which cross-platform things like linking behavior and the ELF format,
and by processor-speciic ABIs (psABI) that define data layout and calling convention.
The gABI can be found at <https://www.sco.com/developers/gabi/>.

#### ARM (Arm32 and AArch64)

ABIs are located at <https://github.com/ARM-software/abi-aa>.
The aapcs32 and aapcs64 directories are likely the most interesting.
These ABIs are actively developed and discussed at that repository.

#### AMDGPU

The LLVM guide at <https://llvm.org/docs/AMDGPUUsage.html> is likely the best collection of
resources for this architecture.

#### AVR

The AVR calling convention is detailed at the GCC Wiki, <https://gcc.gnu.org/wiki/avr-gcc>.
The wiki is actively updated as needed.

#### BPF

A minimal ABI is defined at <https://docs.kernel.org/bpf/standardization/abi.html>.

#### CSky

The LLVM documentation links to <https://github.com/c-sky/csky-doc/tree/master>,
which contains a document `C-SKY_V2_CPU_Applications_Binary_Interface_Standards_Manual.pdf`.

#### Hexagon

The Hexagon ABI can be found at <https://docs.qualcomm.com/doc/80-N2040-23/topic/>.

#### Loongarch

ABIs are located at <https://github.com/loongson/la-abi-specs>, with a rendered version at
<https://loongson.github.io/LoongArch-Documentation/LoongArch-ELF-ABI-EN.html>.
This ABI is actively developed and discussed at that repository.

#### m68k

An ABI is described at <https://m680x0.github.io/doc/abi.html>.

#### MIPS

The o32 ("old 32") ABI is described in [_System V Application Binary Interface: MIPS
RISC Processor Supplement_](https://refspecs.linuxfoundation.org/elf/mipsabi.pdf).
The n32 and n64 ("new 32" and "new 64") ABIs are described in the [_MIPSproTM N32 ABI
Handbook_](https://irix7.com/techpubs/007-2816-005.pdf).
There is also a porting guide for n64 that
may be helpful <https://irix7.com/techpubs/007-2391-006.pdf>.

#### MSP430

The document _MSP430 Embedded Application Binary Interface_ is available at
<https://www.ti.com/lit/an/slaa534a/slaa534a.pdf>.

#### NVPTX

LLVM documentation directs to <https://docs.nvidia.com/cuda/index.html>.

#### PowerPC

There are a few relevant documents for PowerPC ABI:

* [_System V Application Binary Interface: PowerPC Processor
  Supplement_](https://refspecs.linuxfoundation.org/elf/elfspec_ppc.pdf) is the original 32-bit
  PowerPC ABI specification.
* [_Power Architecture™ 32-bit Application Binary Interface Supplement 1.0 - Linux® &
  Embedded_](https://web.archive.org/web/20120608163804/https://www.power.org/resources/downloads/Power-Arch-32-bit-ABI-supp-1.0-Unified.pdf)
  (web archive) is the latest version of the 32-bit ABI.
  Its source is hosted at <https://github.com/ryanarn/powerabi>, but is effectively unmaintained.
* [_64-bit PowerPC ELF Application Binary Interface Supplement
  1.9_](https://refspecs.linuxfoundation.org/ELF/ppc64/PPC-elf64abi-1.9.html) is the original 64-bit
  ABI, last updated in 2004.
  This is known as ELFv1,
  and still used by default on some big endian PowerPC64 targets.
* [_64-bit ELF V2 ABI Specification: Power
  Architecture_](https://openpower.foundation/specifications/64bitelfabi/) is the replacement ABI
  and used by default on most PowerPC64 targets.

#### RISC-V

Ratified RISC-V ABIs are located at <https://docs.riscv.org/reference/abi/v1.0/index.html>.
More recent snapshots can be viewed at <https://riscv-non-isa.github.io/riscv-elf-psabi-doc/>,
with development happening at <https://github.com/riscv-non-isa/riscv-elf-psabi-doc>.

#### s390x

The ABI is located at <https://github.com/IBM/s390x-abi> with a published PDF available at
<https://ibm.github.io/s390x-abi/lzsabi_s390x.pdf>.
This ABI is actively developed and discussed at that repository.

Historical ISAs can be found at <https://linux.mainframe.blog/zarchitecture-principles-of-operation/>.
<https://maskray.me/blog/toolchain-notes-on-z-architecture> has a great overview of other ABI
components such as linking and TLS.

#### SPARC

_SCD 2.4.1_ available at <https://sparc.org/technical-documents/specifications/> describes both the
32-bit and 64-bit ABIs.
_SPARC psABI 3.0_, available at that same link, is an older description of the 32-bit ABI.

#### x86

The 32-bit ABI (i386) is located at <https://gitlab.com/x86-psABIs/i386-ABI>.
The 64-bit ABI (amd64)
is located at <https://gitlab.com/x86-psABIs/x86-64-ABI/#elf-x86-64-abi-psabi>.
Both of these are
actively developed and discussed on mailing lists at <https://groups.google.com/g/ia32-abi> and
<https://groups.google.com/g/x86-64-abi>.

#### Xtensa

TODO

### Apple

Apple largely follows the System V ABIs but documents a few exceptions at the following links:

* Arm64: <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms>
* x86: <https://developer.apple.com/documentation/xcode/writing-64-bit-intel-code-for-apple-platforms>

### Microsoft

Microsoft's x86-64 ABI is defined at <https://learn.microsoft.com/en-us/cpp/build/x64-software-conventions>.
There are a few different x86-32 conventions:

* Cdecl (default): <https://learn.microsoft.com/en-us/cpp/cpp/cdecl>
* Stdcall: <https://learn.microsoft.com/en-us/cpp/cpp/stdcall>
* Fastcall: <https://learn.microsoft.com/en-us/cpp/cpp/fastcall>
* Thiscall: <https://en.wikipedia.org/wiki/X86_calling_conventions#thiscall>

As well as a few that aren't supported by Rust (`__clrcall`, `__vectorcall`, `__preserve_none`).

Aarch64 targets generally use the AAPCS64 calling convention.
There are some exceptions listed at
<https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170>.

Register mappings for arm64ec are documented at
<https://learn.microsoft.com/en-us/cpp/build/arm64ec-windows-abi-conventions?view=msvc-170>.
<http://www.emulators.com/docs/abc_arm64ec_explained.htm> does a great job of explaining more about
how this works.

#### Wasm

The way that C types are broken down into Wasm types is described at
<https://github.com/WebAssembly/tool-conventions/blob/main/BasicCABI.md>.

### Additional Resources for Platform Specifics

The following links aggregate more resources

* Links to other resources such as ELF and DWARF: <https://refspecs.linuxfoundation.org/>
* Links to a number of standards including ISAs <https://llvm.org/docs/CompilerWriterInfo.html>
* More about System V ABIs, especially x86: <https://wiki.osdev.org/System_V_ABI>
