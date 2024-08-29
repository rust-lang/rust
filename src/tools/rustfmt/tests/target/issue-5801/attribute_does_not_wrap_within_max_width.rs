// rustfmt-config: issue-5801-v2.toml

pub enum Severity {
    #[something(AAAAAAAAAAAAA, BBBBBBBBBBBBBB, CCCCCCCCCCCCCCCC, DDDDDDDDDDDDD, EEEEEEEEEEEE, FFFFFFFFFFF, GGGGGGGGGGG)]
    AttrsWillWrap,
    #[something_else(hhhhhhhhhhhhhhhh, iiiiiiiiiiiiiiii, jjjjjjjjjjjjjjj, kkkkkkkkkkkkk, llllllllllll, mmmmmmmmmmmmmm)]
    AttrsWontWrap,
}
