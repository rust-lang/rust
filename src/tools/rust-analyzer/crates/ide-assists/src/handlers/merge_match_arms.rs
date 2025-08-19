use hir::Type;
use ide_db::FxHashMap;
use std::iter::successors;
use syntax::{
    Direction,
    algo::neighbor,
    ast::{self, AstNode, HasName},
};

use crate::{AssistContext, AssistId, Assists, TextRange};

// Assist: merge_match_arms
//
// Merges the current match arm with the following if their bodies are identical.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         $0Action::Move(..) => foo(),
//         Action::Stop => foo(),
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move(..) | Action::Stop => foo(),
//     }
// }
// ```
pub(crate) fn merge_match_arms(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let current_arm = ctx.find_node_at_trimmed_offset::<ast::MatchArm>()?;
    // Don't try to handle arms with guards for now - can add support for this later
    if current_arm.guard().is_some() {
        return None;
    }
    let current_expr = current_arm.expr()?;
    let current_text_range = current_arm.syntax().text_range();
    let current_arm_types = get_arm_types(ctx, &current_arm);
    let multi_arm_selection = !ctx.has_empty_selection()
        && ctx.selection_trimmed().end() > current_arm.syntax().text_range().end();

    // We check if the following match arms match this one. We could, but don't,
    // compare to the previous match arm as well.
    let arms_to_merge = successors(Some(current_arm), |it| neighbor(it, Direction::Next))
        .take_while(|arm| match arm.expr() {
            Some(expr) if arm.guard().is_none() => {
                // don't include match arms that start after our selection
                if multi_arm_selection
                    && arm.syntax().text_range().start() >= ctx.selection_trimmed().end()
                {
                    return false;
                }

                let same_text = expr.syntax().text() == current_expr.syntax().text();
                if !same_text {
                    return false;
                }

                are_same_types(&current_arm_types, arm, ctx)
            }
            _ => false,
        })
        .collect::<Vec<_>>();

    if arms_to_merge.len() <= 1 {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("merge_match_arms"),
        "Merge match arms",
        current_text_range,
        |edit| {
            let pats = if arms_to_merge.iter().any(contains_placeholder) {
                "_".into()
            } else {
                arms_to_merge
                    .iter()
                    .filter_map(ast::MatchArm::pat)
                    .map(|x| x.syntax().to_string())
                    .collect::<Vec<String>>()
                    .join(" | ")
            };

            let arm = format!("{pats} => {current_expr},");

            if let [first, .., last] = &*arms_to_merge {
                let start = first.syntax().text_range().start();
                let end = last.syntax().text_range().end();

                edit.replace(TextRange::new(start, end), arm);
            }
        },
    )
}

fn contains_placeholder(a: &ast::MatchArm) -> bool {
    matches!(a.pat(), Some(ast::Pat::WildcardPat(..)))
}

fn are_same_types(
    current_arm_types: &FxHashMap<String, Option<Type<'_>>>,
    arm: &ast::MatchArm,
    ctx: &AssistContext<'_>,
) -> bool {
    let arm_types = get_arm_types(ctx, arm);
    for (other_arm_type_name, other_arm_type) in arm_types {
        match (current_arm_types.get(&other_arm_type_name), other_arm_type) {
            (Some(Some(current_arm_type)), Some(other_arm_type))
                if other_arm_type == *current_arm_type => {}
            _ => return false,
        }
    }

    true
}

fn get_arm_types<'db>(
    context: &AssistContext<'db>,
    arm: &ast::MatchArm,
) -> FxHashMap<String, Option<Type<'db>>> {
    let mut mapping: FxHashMap<String, Option<Type<'db>>> = FxHashMap::default();

    fn recurse<'db>(
        map: &mut FxHashMap<String, Option<Type<'db>>>,
        ctx: &AssistContext<'db>,
        pat: &Option<ast::Pat>,
    ) {
        if let Some(local_pat) = pat {
            match local_pat {
                ast::Pat::TupleStructPat(tuple) => {
                    for field in tuple.fields() {
                        recurse(map, ctx, &Some(field));
                    }
                }
                ast::Pat::TuplePat(tuple) => {
                    for field in tuple.fields() {
                        recurse(map, ctx, &Some(field));
                    }
                }
                ast::Pat::RecordPat(record) => {
                    if let Some(field_list) = record.record_pat_field_list() {
                        for field in field_list.fields() {
                            recurse(map, ctx, &field.pat());
                        }
                    }
                }
                ast::Pat::ParenPat(parentheses) => {
                    recurse(map, ctx, &parentheses.pat());
                }
                ast::Pat::SlicePat(slice) => {
                    for slice_pat in slice.pats() {
                        recurse(map, ctx, &Some(slice_pat));
                    }
                }
                ast::Pat::IdentPat(ident_pat) => {
                    if let Some(name) = ident_pat.name() {
                        let pat_type = ctx.sema.type_of_binding_in_pat(ident_pat);
                        map.insert(name.text().to_string(), pat_type);
                    }
                }
                _ => (),
            }
        }
    }

    recurse(&mut mapping, context, &arm.pat());
    mapping
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn merge_match_arms_single_patterns() {
        check_assist(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    let x = X::A;
    let y = match x {
        X::A => { 1i32$0 }
        X::B => { 1i32 }
        X::C => { 2i32 }
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    let x = X::A;
    let y = match x {
        X::A | X::B => { 1i32 },
        X::C => { 2i32 }
    }
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_multiple_patterns() {
        check_assist(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X { A, B, C, D, E }

fn main() {
    let x = X::A;
    let y = match x {
        X::A | X::B => {$0 1i32 },
        X::C | X::D => { 1i32 },
        X::E => { 2i32 },
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C, D, E }

fn main() {
    let x = X::A;
    let y = match x {
        X::A | X::B | X::C | X::D => { 1i32 },
        X::E => { 2i32 },
    }
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_placeholder_pattern() {
        check_assist(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X { A, B, C, D, E }

fn main() {
    let x = X::A;
    let y = match x {
        X::A => { 1i32 },
        X::B => { 2i$032 },
        _ => { 2i32 }
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C, D, E }

fn main() {
    let x = X::A;
    let y = match x {
        X::A => { 1i32 },
        _ => { 2i32 },
    }
}
"#,
        );
    }

    #[test]
    fn merges_all_subsequent_arms() {
        check_assist(
            merge_match_arms,
            r#"
enum X { A, B, C, D, E }

fn main() {
    match X::A {
        X::A$0 => 92,
        X::B => 92,
        X::C => 92,
        X::D => 62,
        _ => panic!(),
    }
}
"#,
            r#"
enum X { A, B, C, D, E }

fn main() {
    match X::A {
        X::A | X::B | X::C => 92,
        X::D => 62,
        _ => panic!(),
    }
}
"#,
        )
    }

    #[test]
    fn merge_match_arms_selection_has_leading_whitespace() {
        check_assist(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
    $0    X::A => 0,
        X::B => 0,$0
        X::C => 1,
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
        X::A | X::B => 0,
        X::C => 1,
    }
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_stops_at_end_of_selection() {
        check_assist(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
    $0    X::A => 0,
        X::B => 0,
        $0X::C => 0,
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
        X::A | X::B => 0,
        X::C => 0,
    }
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_works_despite_accidental_selection() {
        check_assist(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
        X::$0A$0 => 0,
        X::B => 0,
        X::C => 1,
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
        X::A | X::B => 0,
        X::C => 1,
    }
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_rejects_guards() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X {
    A(i32),
    B,
    C
}

fn main() {
    let x = X::A;
    let y = match x {
        X::A(a) if a > 5 => { $01i32 },
        X::B => { 1i32 },
        X::C => { 2i32 }
    }
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_different_type() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
//- minicore: result
fn func() {
    match Result::<f64, f32>::Ok(0f64) {
        Ok(x) => $0x.classify(),
        Err(x) => x.classify()
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_different_type_multiple_fields() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
//- minicore: result
fn func() {
    match Result::<(f64, f64), (f32, f32)>::Ok((0f64, 0f64)) {
        Ok(x) => $0x.1.classify(),
        Err(x) => x.1.classify()
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_same_type_multiple_fields() {
        check_assist(
            merge_match_arms,
            r#"
//- minicore: result
fn func() {
    match Result::<(f64, f64), (f64, f64)>::Ok((0f64, 0f64)) {
        Ok(x) => $0x.1.classify(),
        Err(x) => x.1.classify()
    };
}
"#,
            r#"
fn func() {
    match Result::<(f64, f64), (f64, f64)>::Ok((0f64, 0f64)) {
        Ok(x) | Err(x) => x.1.classify(),
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_same_type_subsequent_arm_with_different_type_in_other() {
        check_assist(
            merge_match_arms,
            r#"
enum MyEnum {
    OptionA(f32),
    OptionB(f32),
    OptionC(f64)
}

fn func(e: MyEnum) {
    match e {
        MyEnum::OptionA(x) => $0x.classify(),
        MyEnum::OptionB(x) => x.classify(),
        MyEnum::OptionC(x) => x.classify(),
    };
}
"#,
            r#"
enum MyEnum {
    OptionA(f32),
    OptionB(f32),
    OptionC(f64)
}

fn func(e: MyEnum) {
    match e {
        MyEnum::OptionA(x) | MyEnum::OptionB(x) => x.classify(),
        MyEnum::OptionC(x) => x.classify(),
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_same_type_skip_arm_with_different_type_in_between() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum MyEnum {
    OptionA(f32),
    OptionB(f64),
    OptionC(f32)
}

fn func(e: MyEnum) {
    match e {
        MyEnum::OptionA(x) => $0x.classify(),
        MyEnum::OptionB(x) => x.classify(),
        MyEnum::OptionC(x) => x.classify(),
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_same_type_different_number_of_fields() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
//- minicore: result
fn func() {
    match Result::<(f64, f64, f64), (f64, f64)>::Ok((0f64, 0f64, 0f64)) {
        Ok(x) => $0x.1.classify(),
        Err(x) => x.1.classify()
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_same_destructuring_different_types() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
struct Point {
    x: i32,
    y: i32,
}

fn func() {
    let p = Point { x: 0, y: 7 };

    match p {
        Point { x, y: 0 } => $0"",
        Point { x: 0, y } => "",
        Point { x, y } => "",
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_range() {
        check_assist(
            merge_match_arms,
            r#"
fn func() {
    let x = 'c';

    match x {
        'a'..='j' => $0"",
        'c'..='z' => "",
        _ => "other",
    };
}
"#,
            r#"
fn func() {
    let x = 'c';

    match x {
        'a'..='j' | 'c'..='z' => "",
        _ => "other",
    };
}
"#,
        );
    }

    #[test]
    fn merge_match_arms_enum_without_field() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum MyEnum {
    NoField,
    AField(u8)
}

fn func(x: MyEnum) {
    match x {
        MyEnum::NoField => $0"",
        MyEnum::AField(x) => ""
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_enum_destructuring_different_types() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum MyEnum {
    Move { x: i32, y: i32 },
    Write(String),
}

fn func(x: MyEnum) {
    match x {
        MyEnum::Move { x, y } => $0"",
        MyEnum::Write(text) => "",
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_enum_destructuring_same_types() {
        check_assist(
            merge_match_arms,
            r#"
enum MyEnum {
    Move { x: i32, y: i32 },
    Crawl { x: i32, y: i32 }
}

fn func(x: MyEnum) {
    match x {
        MyEnum::Move { x, y } => $0"",
        MyEnum::Crawl { x, y } => "",
    };
}
        "#,
            r#"
enum MyEnum {
    Move { x: i32, y: i32 },
    Crawl { x: i32, y: i32 }
}

fn func(x: MyEnum) {
    match x {
        MyEnum::Move { x, y } | MyEnum::Crawl { x, y } => "",
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_enum_destructuring_same_types_different_name() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum MyEnum {
    Move { x: i32, y: i32 },
    Crawl { a: i32, b: i32 }
}

fn func(x: MyEnum) {
    match x {
        MyEnum::Move { x, y } => $0"",
        MyEnum::Crawl { a, b } => "",
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_enum_nested_pattern_different_names() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum Color {
    Rgb(i32, i32, i32),
    Hsv(i32, i32, i32),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(Color),
}

fn main(msg: Message) {
    match msg {
        Message::ChangeColor(Color::Rgb(r, g, b)) => $0"",
        Message::ChangeColor(Color::Hsv(h, s, v)) => "",
        _ => "other"
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_enum_nested_pattern_same_names() {
        check_assist(
            merge_match_arms,
            r#"
enum Color {
    Rgb(i32, i32, i32),
    Hsv(i32, i32, i32),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(Color),
}

fn main(msg: Message) {
    match msg {
        Message::ChangeColor(Color::Rgb(a, b, c)) => $0"",
        Message::ChangeColor(Color::Hsv(a, b, c)) => "",
        _ => "other"
    };
}
        "#,
            r#"
enum Color {
    Rgb(i32, i32, i32),
    Hsv(i32, i32, i32),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(Color),
}

fn main(msg: Message) {
    match msg {
        Message::ChangeColor(Color::Rgb(a, b, c)) | Message::ChangeColor(Color::Hsv(a, b, c)) => "",
        _ => "other"
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_enum_destructuring_with_ignore() {
        check_assist(
            merge_match_arms,
            r#"
enum MyEnum {
    Move { x: i32, a: i32 },
    Crawl { x: i32, b: i32 }
}

fn func(x: MyEnum) {
    match x {
        MyEnum::Move { x, .. } => $0"",
        MyEnum::Crawl { x, .. } => "",
    };
}
        "#,
            r#"
enum MyEnum {
    Move { x: i32, a: i32 },
    Crawl { x: i32, b: i32 }
}

fn func(x: MyEnum) {
    match x {
        MyEnum::Move { x, .. } | MyEnum::Crawl { x, .. } => "",
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_nested_with_conflicting_identifier() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum Color {
    Rgb(i32, i32, i32),
    Hsv(i32, i32, i32),
}

enum Message {
    Move { x: i32, y: i32 },
    ChangeColor(u8, Color),
}

fn main(msg: Message) {
    match msg {
        Message::ChangeColor(x, Color::Rgb(y, b, c)) => $0"",
        Message::ChangeColor(y, Color::Hsv(x, b, c)) => "",
        _ => "other"
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_tuple() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
fn func() {
    match (0, "boo") {
        (x, y) => $0"",
        (y, x) => "",
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_parentheses() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
fn func(x: i32) {
    let variable = 2;
    match x {
        1 => $0"",
        ((((variable)))) => "",
        _ => "other"
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_refpat() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
fn func() {
    let name = Some(String::from(""));
    let n = String::from("");
    match name {
        Some(ref n) => $0"",
        Some(n) => "",
        _ => "other",
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_slice() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
fn func(binary: &[u8]) {
    let space = b' ';
    match binary {
        [0x7f, b'E', b'L', b'F', ..] => $0"",
        [space] => "",
        _ => "other",
    };
}
        "#,
        )
    }

    #[test]
    fn merge_match_arms_slice_identical() {
        check_assist(
            merge_match_arms,
            r#"
fn func(binary: &[u8]) {
    let space = b' ';
    match binary {
        [space, 5u8] => $0"",
        [space] => "",
        _ => "other",
    };
}
        "#,
            r#"
fn func(binary: &[u8]) {
    let space = b' ';
    match binary {
        [space, 5u8] | [space] => "",
        _ => "other",
    };
}
        "#,
        )
    }
}
