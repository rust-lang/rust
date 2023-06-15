// run-pass
#![feature(explicit_tail_calls)]

fn main() {
    use Inst::*;

    let program = [
        Inc,       // st + 1 = 1
        ShiftLeft, // st * 2 = 2
        ShiftLeft, // st * 2 = 4
        ShiftLeft, // st * 2 = 8
        Inc,       // st + 1 = 9
        Inc,       // st + 1 = 10
        Inc,       // st + 1 = 11
        ShiftLeft, // st * 2 = 22
        Halt,      // st = 22
    ];

    assert_eq!(run(0, &program), 22);
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Inst {
    Inc,
    ShiftLeft,
    Halt,
}

fn run(state: u32, program: &[Inst]) -> u32 {
    const DISPATCH_TABLE: [fn(u32, &[Inst]) -> u32; 3] = [inc, shift_left, |st, _| st];

    become DISPATCH_TABLE[program[0] as usize](state, program);
}

fn inc(state: u32, program: &[Inst]) -> u32 {
    assert_eq!(program[0], Inst::Inc);
    become run(state + 1, &program[1..])
}

fn shift_left(state: u32, program: &[Inst]) -> u32 {
    assert_eq!(program[0], Inst::ShiftLeft);
    become run(state << 1, &program[1..])
}
