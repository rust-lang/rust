pub mod inner {
    #[cfg(false)]
    pub fn uwu() {}

    #[cfg(false)]
    pub mod cfgd_out {
        pub fn hello() {}
    }

    cfg_select! {
        false => {
            pub mod selected_out {
                pub fn hello() {}
            }
        }
        _ => {}
    }

    pub mod wrong {
        #[cfg(feature = "suggesting me fails the test!!")]
        pub fn meow() {}
    }

    pub mod right {
        #[cfg(feature = "what-a-cool-feature")]
        pub fn meow() {}
    }
}

#[cfg(i_dont_exist_and_you_can_do_nothing_about_it)]
pub fn vanished() {}
