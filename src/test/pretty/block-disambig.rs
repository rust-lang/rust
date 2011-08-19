// Tests that the pretty printer correctly disambiguates various scenarios
// involving block statements by ending them with a semi-colon
fn test1() {
    let val = @0;
    {};
    *val;
}

fn test2() -> int {
    let val = @0;
    {};
    *val
}

fn test3() {
    let regs = @{mutable eax: 0};
    alt true {
      true { }
    };
    (*regs).eax = 1;
}

fn test4() -> bool {
    let regs = @true;
    if true { };
    *regs || false
}

fn test5() -> (int, int) {
    {};
    (0, 1)
}

fn test6() -> bool {
    {};
    (true || false) && true
}

fn test7() -> uint {
    let regs = @0;
    alt true { true { } };
    (*regs < 2) as uint
}

fn test8() -> int {
    let val = @0;
    alt true { true { } };
    *val < 1 ? 0 : 1
}

fn test9() {
    let regs = @mutable 0;
    alt true {
      true { }
    };
    *regs += 1;
}

fn test10() -> int {
    let regs = @mutable [0];
    alt true {
      true { }
    };
    (*regs).(0)
}
