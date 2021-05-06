// check-pass

macro_rules! mac {
    (expr $expr:expr) => {};
    (stmt $stmt:stmt) => {};
}

fn main() {
    mac!(expr #[allow(warnings)] 0);
    mac!(stmt 0);
    mac!(stmt {});
    mac!(stmt path);
    mac!(stmt 0 + 1);
    mac!(stmt path + 1);
}
