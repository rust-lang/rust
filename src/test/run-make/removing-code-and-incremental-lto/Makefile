-include ../../run-make-fulldeps/tools.mk

# How to run this
# $ ./x.py clean
# $ ./x.py test --target thumbv7m-none-eabi src/test/run-make

# The original target of interest was thumbv7em-none-eabi
#
# It is possible that other targets could be tested, but as long as we are
# testing *something*, that is enough for me. (Note that you should not just add
# `# only-target` lines; see rust-lang/rust#67018.)
ifeq (,$(filter $(TARGET),thumbv6m-none-eabi thumbv7em-none-eabi thumbv7em-none-eabihf thumbv7m-none-eabi))

all:

else

# This is a test of a scenario that arose in issue rust-lang/rust#59535.
#
# What the user experienced: deleting a method use caused a link-time failure.
#
# What the cause was: At the time, incremental compilation would determine which
# object files could be reused from a previous build based on, in part, 1. the
# "greenness" of the modules in the current crate (i.e. which modules' contents
# have changed since the incremental build) and 2. the current LTO import
# information of the modules in the current crate.
#
# The problem was that the reused object file could have been optimized based on
# LTO imports of a *previous* compile, not the current one. In other words, the
# past LTO import information could have included more modules than what the
# current LTO imports do, and those dependencies need to be respected when you
# decide what object files to reuse.
#
# To make this more concrete: Here is the the high-level description of the
# specific scenario from rust-lang/rust#59535:
#
# We had a call-graph like this: `[A] -> [B -> D] <- [C]`, where modules are in `[]`
#
# and the change between incremental builds was the `D <- C` link was removed.
#
# 1ST COMPILE: At the time of the first compile, LTO-optimization inlined the
# code from `B` into `A`, so that in the object code for `A`, there was now a
# direct link to the symbol for `D`. The LTO imports computed during this first
# compile showed this directly: it said that `[A]` imported `B` and `D`.
#
# 2ND COMPIILE: But on the second compile, after the developer removed the call
# `D <- C`, the Rust compiler itself determined that the `D` could be declared
# as an internal definition, and then LLVM optimized the definition of `D` away
# entirely. At the *same time*, the LTO imports computed during this second
# compile reported by LLVM said that `[A]` *solely* imported `B`, and so the
# Rust compiler's incremental reuse mechanism determined that we could reuse the
# object code previously generated (during the 1st compile) for `[A]`. This
# conclusion seemed valid to the compiler, because nothing changed about any
# imports that it knew of for `[A]` based on the current compile.

### (Switch to rustup's nightly to observe bug)
# RUSTC := rustc +nightly
RUSTC := $(RUSTC_ORIGINAL)

OUT_DIR = $(TMPDIR)

INCREMENTAL_DIR = $(OUT_DIR)/incr
RUBBLE1_OUT_DIR = $(OUT_DIR)/rubble1
RUBBLE2_OUT_DIR = $(OUT_DIR)/rubble2

LINK_X_DIR := .

all:
	mkdir -p $(OUT_DIR)
	mkdir -p $(INCREMENTAL_DIR)
	mkdir -p $(RUBBLE1_OUT_DIR)
	mkdir -p $(RUBBLE2_OUT_DIR)
	$(RUSTC) --crate-name cortex_m_rt cortex-m-rt.rs --crate-type lib --emit=metadata,link -C opt-level=s --out-dir $(OUT_DIR) --target $(TARGET) -C link-arg=-Tlink.x.in -L $(LINK_X_DIR)
	$(RUSTC) --edition=2018 --crate-name nrf52810_hal nrf52810-hal.rs --crate-type lib --emit=metadata,link -C opt-level=s -C metadata=aa86958b67bf89f5 --out-dir $(OUT_DIR) --target $(TARGET) --extern cortex_m_rt=$(OUT_DIR)/libcortex_m_rt.rmeta -C link-arg=-Tlink.x.in -L $(LINK_X_DIR)
	cp rubble.rs.v1 $(TMPDIR)/rubble.rs
	$(RUSTC) --crate-name rubble $(TMPDIR)/rubble.rs --crate-type bin --emit=link -C opt-level=s --out-dir $(RUBBLE1_OUT_DIR) --target $(TARGET) -C incremental=$(INCREMENTAL_DIR) -L dependency=$(OUT_DIR) --extern nrf52810_hal=$(OUT_DIR)/libnrf52810_hal.rlib -C link-arg=-Tlink.x.in -L $(LINK_X_DIR) -C linker-flavor=ld.lld -C codegen-units=2
	cp rubble.rs.v2 $(TMPDIR)/rubble.rs
	$(RUSTC) --crate-name rubble $(TMPDIR)/rubble.rs --crate-type bin --emit=link -C opt-level=s --out-dir $(RUBBLE2_OUT_DIR) --target $(TARGET) -C incremental=$(INCREMENTAL_DIR) -L dependency=$(OUT_DIR) --extern nrf52810_hal=$(OUT_DIR)/libnrf52810_hal.rlib -C link-arg=-Tlink.x.in -L $(LINK_X_DIR) -C linker-flavor=ld.lld -C codegen-units=2
