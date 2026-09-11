// rustfmt-match_arm_indent: false
// rustfmt-hard_tabs: true
// rustfmt-tab_spaces: 8

// Large-indentation style, brought to you by the Linux kernel
fn foo() {
	match value {
	0 => {
		"one";
		"two";
	}
	1 | 2 | 3 => {
		"line1";
		"line2";
	}
	100..1000 => oneline(),

	_ => {
		// catch-all
		todo!();
	}
	}
}
