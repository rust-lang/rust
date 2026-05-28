// rustfmt-match_arm_indent: false
// Unindent the match arms

fn foo() {
    match x {
        a => {
            "line1";
            "line2"
        }
        ThisIsA::Guard if true => {
            "line1";
            "line2"
        }
        ThisIsA::ReallyLongPattern(ThatWillForce::TheGuard, ToWrapOnto::TheFollowingLine) if true => {
            "line1";
            "line2"
        }
    }
}
