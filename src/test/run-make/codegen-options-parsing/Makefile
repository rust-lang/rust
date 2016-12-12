-include ../tools.mk

all:
	#Option taking a number
	$(RUSTC) -C codegen-units dummy.rs 2>&1 | \
		grep 'codegen option `codegen-units` requires a number'
	$(RUSTC) -C codegen-units= dummy.rs 2>&1 | \
		grep 'incorrect value `` for codegen option `codegen-units` - a number was expected'
	$(RUSTC) -C codegen-units=foo dummy.rs 2>&1 | \
		grep 'incorrect value `foo` for codegen option `codegen-units` - a number was expected'
	$(RUSTC) -C codegen-units=1 dummy.rs
	#Option taking a string
	$(RUSTC) -C extra-filename dummy.rs 2>&1 | \
		grep 'codegen option `extra-filename` requires a string'
	$(RUSTC) -C extra-filename= dummy.rs 2>&1
	$(RUSTC) -C extra-filename=foo dummy.rs 2>&1
	#Option taking no argument
	$(RUSTC) -C lto= dummy.rs 2>&1 | \
		grep 'codegen option `lto` takes no value'
	$(RUSTC) -C lto=1 dummy.rs 2>&1 | \
		grep 'codegen option `lto` takes no value'
	$(RUSTC) -C lto=foo dummy.rs 2>&1 | \
		grep 'codegen option `lto` takes no value'
	$(RUSTC) -C lto dummy.rs

	# Should not link dead code...
	$(RUSTC) -Z print-link-args dummy.rs 2>&1 | \
		grep -e '--gc-sections' -e '-dead_strip' -e '/OPT:REF'
	# ... unless you specifically ask to keep it
	$(RUSTC) -Z print-link-args -C link-dead-code dummy.rs 2>&1 | \
		(! grep -e '--gc-sections' -e '-dead_strip' -e '/OPT:REF')
