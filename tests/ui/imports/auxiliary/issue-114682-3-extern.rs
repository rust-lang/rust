mod gio {
    pub trait SettingsExt {
        fn abc(&self) {}
    }
    impl<T> SettingsExt for T {}
}

mod gtk {
    pub trait SettingsExt {
        fn efg(&self) {}
    }
    impl<T> SettingsExt for T {}
}

pub use gtk::*;
pub use gio::*;
