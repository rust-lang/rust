// rustfmt-match_arm_indent: false
// rustfmt-match_arm_leading_pipes: Always

fn pipes() {
    match value {
        SingleOption => {1;2;}

        Multiple | Options | Together => {1;2;}

        Multiple | Options | But | Long | And | This | Time | With | A | Guard if condition => {1; 2;}

        Multiple | Options | But | Long | And | This | Time | With | A | Guard if condition || and || single_expr => 1,

        a@Enum::Variant1 | a@Enum::Variant2 => {1;2;}

        #[attr] JustInCase => r#final(),
    }
}
