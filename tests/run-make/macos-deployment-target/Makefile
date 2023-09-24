# only-macos
#
# Check that a set deployment target actually makes it to the linker.
# This is important since its a compatibility hazard. The linker will
# generate load commands differently based on what minimum OS it can assume.

include ../tools.mk

ifeq ($(strip $(shell uname -m)),arm64)
	GREP_PATTERN = "minos 11.0"
else
 	GREP_PATTERN = "version 10.13"
endif

OUT_FILE=$(TMPDIR)/with_deployment_target.dylib
all:
	env MACOSX_DEPLOYMENT_TARGET=10.13 $(RUSTC) with_deployment_target.rs -o $(OUT_FILE)
# XXX: The check is for either the x86_64 minimum OR the aarch64 minimum (M1 starts at macOS 11).
# They also use different load commands, so we let that change with each too. The aarch64 check
# isn't as robust as the x86 one, but testing both seems unneeded.
	vtool -show-build $(OUT_FILE) | $(CGREP) -e $(GREP_PATTERN)
