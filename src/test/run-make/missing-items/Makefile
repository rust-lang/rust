-include ../tools.mk

all:
	$(RUSTC) m1.rs -C prefer-dynamic
	$(RUSTC) m2.rs 2>&1 | grep "error\[E0046\]: not all trait items implemented, missing: .*"
	$(RUSTC) m2.rs 2>&1 | grep "  --> m2.rs:18:1"
	$(RUSTC) m2.rs 2>&1 | grep "   | ^ missing .CONSTANT., .Type., .method. in implementation"
	$(RUSTC) m2.rs 2>&1 | grep "   = note: .CONSTANT. from trait: .const CONSTANT: u32;."
	$(RUSTC) m2.rs 2>&1 | grep "   = note: .Type. from trait: .type Type;."
	$(RUSTC) m2.rs 2>&1 | grep "   = note: .method. from trait: .fn(&Self, std::string::String) -> <Self as m1::X>::Type."
