-include ../tools.mk

TARGETS =
ifeq ($(filter arm,$(LLVM_COMPONENTS)),arm)
# construct a fairly exhaustive list of platforms that we
# support. These ones don't follow a pattern
TARGETS += arm-linux-androideabi arm-unknown-linux-gnueabihf arm-unknown-linux-gnueabi
endif

ifeq ($(filter x86,$(LLVM_COMPONENTS)),x86)
X86_ARCHS = i686 x86_64
else
X86_ARCHS =
endif

# these ones do, each OS lists the architectures it supports
LINUX=$(filter aarch64 mips,$(LLVM_COMPONENTS)) $(X86_ARCHS)
ifeq ($(filter mips,$(LLVM_COMPONENTS)),mips)
LINUX += mipsel
endif

WINDOWS=$(X86_ARCHS)
# fails with: failed to get iphonesimulator SDK path: no such file or directory
#IOS=i386 aarch64 armv7
DARWIN=$(X86_ARCHS)

$(foreach arch,$(LINUX),$(eval TARGETS += $(arch)-unknown-linux-gnu))
$(foreach arch,$(WINDOWS),$(eval TARGETS += $(arch)-pc-windows-gnu))
#$(foreach arch,$(IOS),$(eval TARGETS += $(arch)-apple-ios))
$(foreach arch,$(DARWIN),$(eval TARGETS += $(arch)-apple-darwin))

all: $(TARGETS)

define MK_TARGETS
# compile the rust file to the given target, but only to asm and IR
# form, to avoid having to have an appropriate linker.
#
# we need some features because the integer SIMD instructions are not
# enabled by-default for i686 and ARM; these features will be invalid
# on some platforms, but LLVM just prints a warning so that's fine for
# now.
$(1): simd.rs
	$$(RUSTC) --target=$(1) --emit=llvm-ir,asm simd.rs \
                -C target-feature='+neon,+sse2' -C extra-filename=-$(1)
endef

$(foreach targetxxx,$(TARGETS),$(eval $(call MK_TARGETS,$(targetxxx))))
