// rustfmt-format_strings: true
// rustfmt-error_on_line_overflow: false

fn main() {
    println!(
        "ThisIsAReallyLongStringWithNoSpaces.It_should_prefer_to_break_onpunctuation:\
         Likethisssssssssssss"
    );
    format!("{}__{}__{}ItShouldOnlyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyNoticeSemicolonsPeriodsColonsAndCommasAndResortToMid-CharBreaksAfterPunctuation{}{}",x,y,z,a,b);
    println!(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaalhijalfhiigjapdighjapdigjapdighdapighapdighpaidhg;\
         adopgihadoguaadbadgad,qeoihapethae8t0aet8haetadbjtaeg;\
         ooeouthaoeutgadlgajduabgoiuadogabudogubaodugbadgadgadga;adoughaoeugbaouea"
    );
}
