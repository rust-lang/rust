#![feature(rustc_attrs)]
#![pattern_complexity_limit = "61"]

//@ check-pass
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
    field19: bool,
    field20: bool,
    field21: bool,
    field22: bool,
    field23: bool,
    field24: bool,
    field25: bool,
    field26: bool,
    field27: bool,
    field28: bool,
    field29: bool,
    field30: bool,
}

fn request_key(command: BaseCommand) {
    match command {
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
        BaseCommand { field19: true, .. } => {}
        BaseCommand { field20: true, .. } => {}
        BaseCommand { field21: true, .. } => {}
        BaseCommand { field22: true, .. } => {}
        BaseCommand { field23: true, .. } => {}
        BaseCommand { field24: true, .. } => {}
        BaseCommand { field25: true, .. } => {}
        BaseCommand { field26: true, .. } => {}
        BaseCommand { field27: true, .. } => {}
        BaseCommand { field28: true, .. } => {}
        BaseCommand { field29: true, .. } => {}
        BaseCommand { field30: true, .. } => {}

        _ => {}
    }
}

fn main() {}
