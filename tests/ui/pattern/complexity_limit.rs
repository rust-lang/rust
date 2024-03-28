//@ check-pass

#![feature(rustc_attrs)]
// By default the complexity is 10_000_000, but we reduce it so the test takes
// (much) less time.
#![pattern_complexity = "2000000"]

#[derive(Default)]
struct BaseCommand {
    field01: bool,
    field02: bool,
    field03: bool,
    field04: bool,
    field05: bool,
    field06: bool,
    field07: bool,
    field08: bool,
    field09: bool,
    field10: bool,
    field11: bool,
    field12: bool,
    field13: bool,
    field14: bool,
    field15: bool,
    field16: bool,
    field17: bool,
    field18: bool,
}

fn request_key(command: BaseCommand) {
    match command { //~ WARN: this pattern-match expression is taking a long time to analyze
        BaseCommand { field01: true, .. } => {}
        BaseCommand { field02: true, .. } => {}
        BaseCommand { field03: true, .. } => {}
        BaseCommand { field04: true, .. } => {}
        BaseCommand { field05: true, .. } => {}
        BaseCommand { field06: true, .. } => {}
        BaseCommand { field07: true, .. } => {}
        BaseCommand { field08: true, .. } => {}
        BaseCommand { field09: true, .. } => {}
        BaseCommand { field10: true, .. } => {}
        BaseCommand { field11: true, .. } => {}
        BaseCommand { field12: true, .. } => {}
        BaseCommand { field13: true, .. } => {}
        BaseCommand { field14: true, .. } => {}
        BaseCommand { field15: true, .. } => {}
        BaseCommand { field16: true, .. } => {}
        BaseCommand { field17: true, .. } => {}
        BaseCommand { field18: true, .. } => {}

        BaseCommand { field01: false, .. } => {}
        BaseCommand { field02: false, .. } => {}
        BaseCommand { field03: false, .. } => {}
        BaseCommand { field04: false, .. } => {}
        BaseCommand { field05: false, .. } => {}
        BaseCommand { field06: false, .. } => {}
        BaseCommand { field07: false, .. } => {}
        BaseCommand { field08: false, .. } => {}
        BaseCommand { field09: false, .. } => {}
        BaseCommand { field10: false, .. } => {}
        BaseCommand { field11: false, .. } => {}
        BaseCommand { field12: false, .. } => {}
        BaseCommand { field13: false, .. } => {}
        BaseCommand { field14: false, .. } => {}
        BaseCommand { field15: false, .. } => {}
        BaseCommand { field16: false, .. } => {}
        BaseCommand { field17: false, .. } => {}
        BaseCommand { field18: false, .. } => {}
    }
}

fn main() {
    request_key(BaseCommand::default());
}
