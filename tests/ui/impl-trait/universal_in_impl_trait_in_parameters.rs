//@ run-pass

use std::fmt::Display;

fn check_display_eq(iter: impl IntoIterator<Item = impl Display>) {
    let mut collected = String::new();
    for it in iter {
        let disp = format!("{} ", it);
        collected.push_str(&disp);
    }
    assert_eq!("0 3 27 823 4891 1 0", collected.trim());
}

fn main() {
    let i32_list = [0i32, 3, 27, 823, 4891, 1, 0];
    let i32_list_vec = vec![0i32, 3, 27, 823, 4891, 1, 0];
    let u32_list = [0u32, 3, 27, 823, 4891, 1, 0];
    let u32_list_vec = vec![0u32, 3, 27, 823, 4891, 1, 0];
    let u16_list = [0u16, 3, 27, 823, 4891, 1, 0];
    let str_list = ["0", "3", "27", "823", "4891", "1", "0"];
    let str_list_vec = vec!["0", "3", "27", "823", "4891", "1", "0"];

    check_display_eq(&i32_list);
    check_display_eq(i32_list_vec);
    check_display_eq(&u32_list);
    check_display_eq(u32_list_vec);
    check_display_eq(&u16_list);
    check_display_eq(&str_list);
    check_display_eq(str_list_vec);
}
