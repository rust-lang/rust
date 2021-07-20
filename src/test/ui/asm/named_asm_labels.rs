#![feature(asm, global_asm)]

fn main() {
    unsafe {
        // Basic usage
        asm!("bar: nop"); //~ ERROR do not use named labels

        // No following asm
        asm!("abcd:"); //~ ERROR do not use named labels

        // Multiple labels on one line
        asm!("foo: bar1: nop");
        //~^ ERROR do not use named labels
        //~| ERROR do not use named labels

        // Multiple lines
        asm!("foo1: nop", "nop"); //~ ERROR do not use named labels
        asm!("foo2: foo3: nop", "nop");
        //~^ ERROR do not use named labels
        //~| ERROR do not use named labels
        asm!("nop", "foo4: nop"); //~ ERROR do not use named labels
        asm!("foo5: nop", "foo6: nop");
        //~^ ERROR do not use named labels
        //~| ERROR do not use named labels

        // Statement separator
        asm!("foo7: nop; foo8: nop");
        //~^ ERROR do not use named labels
        //~| ERROR do not use named labels
        asm!("foo9: nop; nop"); //~ ERROR do not use named labels
        asm!("nop; foo10: nop"); //~ ERROR do not use named labels

        // Escaped newline
        asm!("bar2: nop\n bar3: nop");
        //~^ ERROR do not use named labels
        //~| ERROR do not use named labels
        asm!("bar4: nop\n nop"); //~ ERROR do not use named labels
        asm!("nop\n bar5: nop"); //~ ERROR do not use named labels
        asm!("nop\n bar6: bar7: nop");
        //~^ ERROR do not use named labels
        //~| ERROR do not use named labels

        // Raw strings
        asm!(
            r"
            blah2: nop
            blah3: nop
            "
        );
        //~^^^^ ERROR do not use named labels
        //~^^^^ ERROR do not use named labels
        asm!(
            r###"
            nop
            nop ; blah4: nop
            "###
        );
        //~^^^ ERROR do not use named labels

        // Non-labels
        // should not trigger lint, but may be invalid asm
        asm!("ab cd: nop");

        // Only `blah:` should trigger
        asm!("1bar: blah: nop"); //~ ERROR do not use named labels

        // Only `blah1:` should trigger
        asm!("blah1: 2bar: nop"); //~ ERROR do not use named labels

        // Duplicate labels
        asm!("def: def: nop"); //~ ERROR do not use named labels
        asm!("def: nop\ndef: nop"); //~ ERROR do not use named labels
        asm!("def: nop; def: nop"); //~ ERROR do not use named labels

        // Trying to break parsing
        asm!(":");
        asm!("\n:\n");
        asm!("::::");

        // 0x3A is a ':'
        asm!("fooo\u{003A} nop"); //~ ERROR do not use named labels
        asm!("foooo\x3A nop"); //~ ERROR do not use named labels

        // 0x0A is a newline
        asm!("fooooo:\u{000A} nop"); //~ ERROR do not use named labels
        asm!("foooooo:\x0A nop"); //~ ERROR do not use named labels

        // Intentionally breaking span finding
        // equivalent to "ABC: nop"
        asm!("\x41\x42\x43\x3A\x20\x6E\x6F\x70"); //~ ERROR do not use named labels
    }
}

// Don't trigger on global asm
global_asm!("aaaaaaaa: nop");
