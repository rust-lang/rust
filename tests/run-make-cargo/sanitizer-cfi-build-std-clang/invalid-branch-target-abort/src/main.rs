// This example demonstrates redirecting control flow using an indirect
// branch/call to an invalid destination (i.e., within the body of the
// function).

use std::mem;

fn add_one(x: i32) -> i32 {
    x + 1
}

#[unsafe(naked)]
pub extern "C" fn add_two(_x: i32) -> ! {
    // x + 2 preceded by a landing pad/nop block
    core::arch::naked_asm!(
        r#"
        nop
        nop
        nop
        nop
        nop
        nop
        nop
        nop
        nop
        lea eax, [rdi + 2]
        ret
        "#,
    );
}

fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {
    f(arg) + f(arg)
}

fn main() {
    let answer = do_twice(add_one, 5);

    println!("The answer is: {}", answer);

    println!("With CFI enabled, you should not see the next answer");
    let f: fn(i32) -> i32 = unsafe {
        // Offset 0 is a valid branch/call destination (i.e., the function entry
        // point), but offsets 1-8 within the landing pad/nop block are invalid
        // branch/call destinations (i.e., within the body of the function).
        mem::transmute::<*const u8, fn(i32) -> i32>((add_two as *const u8).offset(5))
    };
    let next_answer = do_twice(f, 5);

    println!("The next answer is: {}", next_answer);
}
