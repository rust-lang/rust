#!/bin/sh

case $1 in
--version) echo  4.0.1;;
--prefix) echo  $SCRATCH/haiku-cross/sysroot/boot/system;;
--bindir) echo  $SCRATCH/haiku-cross/sysroot/boot/system/bin;;
--includedir) echo  $SCRATCH/haiku-cross/sysroot/boot/system/develop/headers;;
--libdir) echo  $SCRATCH/haiku-/cross/sysroot/boot/system/develop/lib;;
--cmakedir) echo  $SCRATCH/haiku-/cross/sysroot/boot/system/develop/lib/cmake/llvm;;
--cppflags) echo  -I$SCRATCH/haiku-/cross/sysroot/boot/system/develop/headers \
                  -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS;;
--cflags) echo  -I$SCRATCH/haiku-cross/sysroot/boot/system/develop/headers \
                -fPIC -Wall -W -Wno-unused-parameter -Wwrite-strings \
                -Wno-missing-field-initializers -pedantic -Wno-long-long -Wno-comment \
                -Werror=date-time -ffunction-sections -fdata-sections -O3 -DNDEBUG \
                -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS;;
--cxxflags) echo  -I/$SCRATCH/haiku-cross/sysroot/boot/system/develop/headers \
                  -fPIC -fvisibility-inlines-hidden -Wall -W -Wno-unused-parameter \
                  -Wwrite-strings -Wcast-qual -Wno-missing-field-initializers -pedantic \
                  -Wno-long-long -Wno-maybe-uninitialized -Wdelete-non-virtual-dtor \
                  -Wno-comment -Werror=date-time -std=c++11 -ffunction-sections \
                  -fdata-sections -O3 -DNDEBUG  -fno-exceptions \
                  -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS;;
--ldflags) echo  -L$SCRATCH/haiku-cross/sysroot/boot/system/develop/lib ;;
--system-libs) echo ;;
--libs) echo  -lLLVM-4.0;;
--libfiles) echo  $SCRATCH/haiku-cross/sysroot/boot/system/develop/lib/libLLVM-4.0.so;;
--components) echo  aarch64 aarch64asmparser aarch64asmprinter aarch64codegen \
                    aarch64desc aarch64disassembler aarch64info aarch64utils all \
                    all-targets amdgpu amdgpuasmparser amdgpuasmprinter amdgpucodegen \
                    amdgpudesc amdgpudisassembler amdgpuinfo amdgpuutils analysis arm \
                    armasmparser armasmprinter armcodegen armdesc armdisassembler \
                    arminfo asmparser asmprinter bitreader bitwriter bpf bpfasmprinter \
                    bpfcodegen bpfdesc bpfdisassembler bpfinfo codegen core coroutines \
                    coverage debuginfocodeview debuginfodwarf debuginfomsf debuginfopdb \
                    demangle engine executionengine globalisel hexagon hexagonasmparser \
                    hexagoncodegen hexagondesc hexagondisassembler hexagoninfo \
                    instcombine instrumentation interpreter ipo irreader lanai \
                    lanaiasmparser lanaicodegen lanaidesc lanaidisassembler lanaiinfo \
                    lanaiinstprinter libdriver lineeditor linker lto mc mcdisassembler \
                    mcjit mcparser mips mipsasmparser mipsasmprinter mipscodegen \
                    mipsdesc mipsdisassembler mipsinfo mirparser msp430 msp430asmprinter \
                    msp430codegen msp430desc msp430info native nativecodegen nvptx \
                    nvptxasmprinter nvptxcodegen nvptxdesc nvptxinfo objcarcopts object \
                    objectyaml option orcjit passes powerpc powerpcasmparser \
                    powerpcasmprinter powerpccodegen powerpcdesc powerpcdisassembler \
                    powerpcinfo profiledata riscv riscvcodegen riscvdesc riscvinfo \
                    runtimedyld scalaropts selectiondag sparc sparcasmparser \
                    sparcasmprinter sparccodegen sparcdesc sparcdisassembler sparcinfo \
                    support symbolize systemz systemzasmparser systemzasmprinter \
                    systemzcodegen systemzdesc systemzdisassembler systemzinfo tablegen \
                    target transformutils vectorize x86 x86asmparser x86asmprinter \
                    x86codegen x86desc x86disassembler x86info x86utils xcore \
                    xcoreasmprinter xcorecodegen xcoredesc xcoredisassembler xcoreinfo;;
--host-target) echo  x86_64-unknown-haiku;;
--has-rtti) echo  YES;;
--shared-mode) echo  shared;;
esac
