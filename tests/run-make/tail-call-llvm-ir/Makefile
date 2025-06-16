# This test verifies that the `become` keyword generates tail call instructions in LLVM IR

# Use the Rust compiler from the build directory
RUSTC ?= ../../../build/aarch64-apple-darwin/stage1/bin/rustc

all: verify-tail-call

verify-tail-call:
	# Create test file with debuginfo to ensure function names are preserved
	$(RUSTC) tail-call-test.rs --emit=llvm-ir -g -C opt-level=0
	
	# Check that `become` generates 'tail call' instructions for with_tail function
	grep -q "tail call.*@_ZN14tail_call_test9with_tail" tail-call-test.ll || (echo "ERROR: 'tail call' instruction not found for with_tail"; exit 1)
	
	# Check that regular recursive calls don't use tail call optimization by default
	grep -q "@_ZN14tail_call_test7no_tail" tail-call-test.ll || (echo "ERROR: no_tail function not found"; exit 1)
	! grep -q "tail call.*@_ZN14tail_call_test7no_tail" tail-call-test.ll || (echo "ERROR: Regular function call incorrectly marked as tail call"; exit 1)
	
	# Check mutual recursion with 'become'
	grep -q "tail call.*@_ZN14tail_call_test14even_with_tail" tail-call-test.ll || (echo "ERROR: 'tail call' instruction not found for even_with_tail"; exit 1)
	grep -q "tail call.*@_ZN14tail_call_test13odd_with_tail" tail-call-test.ll || (echo "ERROR: 'tail call' instruction not found for odd_with_tail"; exit 1)
	
	# The test passes only if all checks above confirm that:
	# 1. Functions using 'become' generate LLVM tail call instructions
	# 2. Regular recursive functions don't generate tail call instructions
	@echo "LLVM IR verification successful: 'become' correctly generates 'tail call' instructions"