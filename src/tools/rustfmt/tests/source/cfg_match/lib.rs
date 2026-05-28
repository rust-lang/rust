#![feature(cfg_match)]

std::cfg_match! {
    test => {
        mod format_me_please_1;
    }
    target_family = "unix" => {
        mod format_me_please_2;
    }
    cfg(target_pointer_width = "32") => {
        mod format_me_please_3;
    }
    _ => {
        mod format_me_please_4;
    }
}
