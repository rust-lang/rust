#![feature(rustc_attrs)]
#![pattern_complexity_limit = "10000"]

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
    match command { //~ ERROR: reached pattern complexity limit
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
        BaseCommand { field19: false, .. } => {}
        BaseCommand { field20: false, .. } => {}
        BaseCommand { field21: false, .. } => {}
        BaseCommand { field22: false, .. } => {}
        BaseCommand { field23: false, .. } => {}
        BaseCommand { field24: false, .. } => {}
        BaseCommand { field25: false, .. } => {}
        BaseCommand { field26: false, .. } => {}
        BaseCommand { field27: false, .. } => {}
        BaseCommand { field28: false, .. } => {}
        BaseCommand { field29: false, .. } => {}
        BaseCommand { field30: false, .. } => {}
    }
}

fn main() {
    request_key(BaseCommand::default());
}
