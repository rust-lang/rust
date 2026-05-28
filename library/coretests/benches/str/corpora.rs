//! Exposes a number of modules with different kinds of strings.
//!
//! Each module contains `&str` constants named `TINY`, `SMALL`, `MEDIUM`,
//! `LARGE`, and `HUGE`.
//!
//! - The `TINY` string is generally around 8 bytes.
//! - The `SMALL` string is generally around 30-40 bytes.
//! - The `MEDIUM` string is generally around 600-700 bytes.
//! - The `LARGE` string is the `MEDIUM` string repeated 8x, and is around 5kb.
//! - The `HUGE` string is the `LARGE` string repeated 8x (or the `MEDIUM`
//!   string repeated 64x), and is around 40kb.
//!
//! Except for `mod emoji` (which is just a bunch of emoji), the strings were
//! pulled from (localizations of) rust-lang.org.

macro_rules! repeat8 {
    ($s:expr) => {
        concat!($s, $s, $s, $s, $s, $s, $s, $s)
    };
}

macro_rules! define_consts {
    ($s:literal) => {
        pub const MEDIUM: &str = $s;
        pub const LARGE: &str = repeat8!($s);
        pub const HUGE: &str = repeat8!(repeat8!(repeat8!($s)));
    };
}

pub mod en {
    pub const TINY: &str = "Mary had";
    pub const SMALL: &str = "Mary had a little lamb, Little lamb";
    define_consts! {
        "Rust is blazingly fast and memory-efficient: with no runtime or garbage
         collector, it can power performance-critical services, run on embedded
         devices, and easily integrate with other languages.  Rust’s rich type system
         and ownership model guarantee memory-safety and thread-safety — enabling you
         to eliminate many classes of bugs at compile-time.  Rust has great
         documentation, a friendly compiler with useful error messages, and top-notch
         tooling — an integrated package manager and build tool, smart multi-editor
         support with auto-completion and type inspections, an auto-formatter, and
         more."
    }
}

pub mod zh {
    pub const TINY: &str = "速度惊";
    pub const SMALL: &str = "速度惊人且内存利用率极高";
    define_consts! {
        "Rust   速度惊人且内存利用率极高。由于\
         没有运行时和垃圾回收，它能够胜任对性能要\
         求特别高的服务，可以在嵌入式设备上运行，\
         还能轻松和其他语言集成。Rust 丰富的类型\
         系统和所有权模型保证了内存安全和线程安全，\
         让您在编译期就能够消除各种各样的错误。\
         Rust 拥有出色的文档、友好的编译器和清晰\
         的错误提示信息， 还集成了一流的工具——\
         包管理器和构建工具， 智能地自动补全和类\
         型检验的多编辑器支持， 以及自动格式化代\
         码等等。"
    }
}

pub mod ru {
    pub const TINY: &str = "Сотни";
    pub const SMALL: &str = "Сотни компаний по";
    define_consts! {
        "Сотни компаний по всему миру используют Rust в реальных\
         проектах для быстрых кросс-платформенных решений с\
         ограниченными ресурсами. Такие проекты, как Firefox,\
         Dropbox и Cloudflare, используют Rust. Rust отлично\
         подходит как для стартапов, так и для больших компаний,\
         как для встраиваемых устройств, так и для масштабируемых\
         web-сервисов. Мой самый большой комплимент Rust."
    }
}

pub mod emoji {
    pub const TINY: &str = "😀😃";
    pub const SMALL: &str = "😀😃😄😁😆😅🤣😂🙂🙃😉😊😇🥰😍🤩😘";
    define_consts! {
        "😀😃😄😁😆😅🤣😂🙂🙃😉😊😇🥰😍🤩😘😗☺😚😙🥲😋😛😜🤪😝🤑🤗🤭🤫🤔🤐🤨😐😑😶😶‍🌫️😏😒\
         🙄😬😮‍💨🤥😌😔😪🤤😴😷🤒🤕🤢🤮🤧🥵🥶🥴😵😵‍💫🤯��🥳🥸😎🤓🧐😕😟🙁☹😮😯😲😳🥺😦😧😨\
         😰😥😢😭😱😖😣😞😓😩😫🥱😤😡😠🤬😈👿💀☠💩🤡👹👺👻👽👾🤖😺😸😹😻😼😽🙀😿😾🙈🙉🙊\
         💋💌💘💝💖💗💓��💕💟❣💔❤️‍🔥❤️‍🩹❤🧡💛💚💙💜🤎🖤🤍💯💢💥💫💦💨🕳💬👁️‍🗨️🗨🗯💭💤👋\
         🤚🖐✋🖖👌🤌🤏✌"
    }
}
