//@ rustfmt-control_brace_style: AlwaysNextLine
//@ rustfmt-match_arm_blocks: false

fn main() {
    let fooooooo = "100000000000000000000000";
    let _bar = match fooooooo
    {
        "100000000000000000000000" =>
            fooooooo.len() == 1 && fooooooo.contains("222222222222222222"),
        _ => unreachable!("Should not happen"),
    };
}
