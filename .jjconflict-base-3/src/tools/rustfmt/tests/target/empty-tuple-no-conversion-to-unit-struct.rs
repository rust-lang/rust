enum TestEnum {
    Arm1(),
    Arm2,
}

fn foo() {
    let test = TestEnum::Arm1;
    match test {
        TestEnum::Arm1() => {}
        TestEnum::Arm2 => {}
    }
}
