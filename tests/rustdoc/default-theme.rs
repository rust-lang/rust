//@ compile-flags: --default-theme ayu

//@ has default_theme/index.html
//@ has - '//script[@id="default-settings"]/@data-theme' 'ayu'
//@ has - '//script[@id="default-settings"]/@data-use_system_theme' 'false'

pub fn whatever() {}
