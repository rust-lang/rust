// rustfmt-style_edition: 2024
pub enum TestEnum {
    a,
    b,
}

fn the_test(input: TestEnum) {
    match input {
        TestEnum::a => String::from("aaa"),
        TestEnum::b => String::from("this is a very very very very very very very very very very very very very very very ong string"),
        
    };
}
