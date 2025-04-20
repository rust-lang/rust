//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

// Tests that the use of named labels in the `asm!` macro are linted against
// except for in `#[naked]` fns.
// Using a named label is incorrect as per the RFC because for most cases
// the compiler cannot ensure that inline asm is emitted exactly once per
// codegen unit (except for naked fns) and so the label could be duplicated
// which causes less readable LLVM errors and in the worst cases causes ICEs
// or segfaults based on system dependent behavior and codegen flags.

use std::arch::{asm, global_asm, naked_asm};

#[no_mangle]
pub static FOO: usize = 42;

fn main() {
    unsafe {
        // Basic usage
        asm!("bar: nop"); //~ ERROR avoid using named labels

        // No following asm
        asm!("abcd:"); //~ ERROR avoid using named labels

        // Multiple labels on one line
        asm!("foo: bar1: nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels

        // Multiple lines
        asm!("foo1: nop", "nop"); //~ ERROR avoid using named labels
        asm!("foo2: foo3: nop", "nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels
        asm!("nop", "foo4: nop"); //~ ERROR avoid using named labels
        asm!("foo5: nop", "foo6: nop");
        //~^ ERROR avoid using named labels
        //~| ERROR avoid using named labels

        // Statement separator
        asm!("foo7: nop; foo8: nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels
        asm!("foo9: nop; nop"); //~ ERROR avoid using named labels
        asm!("nop; foo10: nop"); //~ ERROR avoid using named labels

        // Escaped newline
        asm!("bar2: nop\n bar3: nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels
        asm!("bar4: nop\n nop"); //~ ERROR avoid using named labels
        asm!("nop\n bar5: nop"); //~ ERROR avoid using named labels
        asm!("nop\n bar6: bar7: nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels

        // Raw strings
        asm!(
            r"
            blah2: nop
            blah3: nop
            "
        );
        //~^^^^ ERROR avoid using named labels
        //~^^^^ ERROR avoid using named labels

        asm!(
            r###"
            nop
            nop ; blah4: nop
            "###
        );
        //~^^^ ERROR avoid using named labels

        // Non-labels
        // should not trigger lint, but may be invalid asm
        asm!("ab cd: nop");

        // `blah:` does not trigger because labels need to be at the start
        // of the statement, and there was already a non-label
        asm!("1bar: blah: nop");

        // Only `blah1:` should trigger
        asm!("blah1: 2bar: nop"); //~ ERROR avoid using named labels

        // Duplicate labels
        asm!("def: def: nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels
        asm!("def: nop\ndef: nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels
        asm!("def: nop; def: nop");
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels

        // Trying to break parsing
        asm!(":");
        asm!("\n:\n");
        asm!("::::");

        // 0x3A is a ':'
        asm!("fooo\u{003A} nop"); //~ ERROR avoid using named labels
        asm!("foooo\x3A nop"); //~ ERROR avoid using named labels

        // 0x0A is a newline
        asm!("fooooo:\u{000A} nop"); //~ ERROR avoid using named labels
        asm!("foooooo:\x0A nop"); //~ ERROR avoid using named labels

        // Intentionally breaking span finding
        // equivalent to "ABC: nop"
        asm!("\x41\x42\x43\x3A\x20\x6E\x6F\x70"); //~ ERROR avoid using named labels

        // Non-label colons - should pass
        asm!("mov rax, qword ptr fs:[0]");

        // Comments
        asm!(
            r"
            ab: nop // ab: does foo
            // cd: nop
            "
        );
        //~^^^^ ERROR avoid using named labels

        // Tests usage of colons in non-label positions
        asm!(":lo12:FOO"); // this is apparently valid aarch64

        // is there an example that is valid x86 for this test?
        asm!(":bbb nop");

        // non-ascii characters are not allowed in labels, so should not trigger the lint
        asm!("Ù: nop");
        asm!("testÙ: nop");
        asm!("_Ù_: nop");

        // Format arguments should be conservatively assumed to be valid characters in labels
        // Would emit `test_rax:` or similar
        #[allow(asm_sub_register)]
        {
            asm!("test_{}: nop", in(reg) 10); //~ ERROR avoid using named labels
        }
        asm!("test_{}: nop", const 10); //~ ERROR avoid using named labels
        asm!("test_{}: nop", sym main); //~ ERROR avoid using named labels
        asm!("{}_test: nop", const 10); //~ ERROR avoid using named labels
        asm!("test_{}_test: nop", const 10); //~ ERROR avoid using named labels
        asm!("{}: nop", const 10); //~ ERROR avoid using named labels

        asm!("{uwu}: nop", uwu = const 10); //~ ERROR avoid using named labels
        asm!("{0}: nop", const 10); //~ ERROR avoid using named labels
        asm!("{1}: nop", "/* {0} */", const 10, const 20); //~ ERROR avoid using named labels

        // Test include_str in asm
        asm!(include_str!("named-asm-labels.s"));
        //~^ ERROR avoid using named labels
        //~^^ ERROR avoid using named labels
        //~^^^ ERROR avoid using named labels
        //~^^^^ ERROR avoid using named labels

        // Test allowing or warning on the lint instead
        #[allow(named_asm_labels)]
        {
            asm!("allowed: nop"); // Should not emit anything
        }

        #[warn(named_asm_labels)]
        {
            asm!("warned: nop"); //~ WARNING avoid using named labels
        }
    }
}

// Trigger on naked fns too, even though they can't be inlined, reusing a
// label or LTO can cause labels to break
#[unsafe(naked)]
pub extern "C" fn foo() -> i32 {
    naked_asm!(".Lfoo: mov rax, {}; ret;", "nop", const 1)
    //~^ ERROR avoid using named labels
}

// Make sure that non-naked attributes *do* still let the lint happen
#[no_mangle]
pub extern "C" fn bar() {
    unsafe { asm!(".Lbar: mov rax, {}; ret;", "nop", const 1, options(noreturn)) }
    //~^ ERROR avoid using named labels
}

#[unsafe(naked)]
pub extern "C" fn aaa() {
    fn _local() {}

    naked_asm!(".Laaa: nop; ret;") //~ ERROR avoid using named labels
}

pub fn normal() {
    fn _local1() {}

    #[unsafe(naked)]
    pub extern "C" fn bbb() {
        fn _very_local() {}

        naked_asm!(".Lbbb: nop; ret;") //~ ERROR avoid using named labels
    }

    fn _local2() {}
}

// Make sure that the lint happens within closures
fn closures() {
    || unsafe {
        asm!("closure1: nop"); //~ ERROR avoid using named labels
    };

    move || unsafe {
        asm!("closure2: nop"); //~ ERROR avoid using named labels
    };

    || {
        #[unsafe(naked)]
        extern "C" fn _nested() {
            naked_asm!("ret;");
        }

        unsafe {
            asm!("closure3: nop"); //~ ERROR avoid using named labels
        }
    };
}

// Don't trigger on global asm
global_asm!("aaaaaaaa: nop");
