-include ../tools.mk

# Testing that the squiggles rustc produces look right, by comparing to some
# drawn in the source file.

# This is what rustc's error messages line looks like for this test:
# squiggle-test.rs:15:11: 15:32 [error begins here]
# ^----------------------------^ We need to compensate for this much whitespace.

OUT=$(shell $(RUSTC) squiggle-test.rs 2>&1 | grep "~" | cut -b 1-30 --complement )
EXPECTED=$(shell cat squiggle-test.rs | grep "~" | cut -b 1-10 --complement )
all:
	echo OUT "$(OUT)"
	echo EXP "$(EXPECTED)"
	test "$(OUT)" = "$(EXPECTED)"
