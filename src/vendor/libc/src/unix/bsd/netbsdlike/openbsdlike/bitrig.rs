s! {
    pub struct lconv {
        pub decimal_point: *mut ::c_char,
        pub thousands_sep: *mut ::c_char,
        pub grouping: *mut ::c_char,
        pub int_curr_symbol: *mut ::c_char,
        pub currency_symbol: *mut ::c_char,
        pub mon_decimal_point: *mut ::c_char,
        pub mon_thousands_sep: *mut ::c_char,
        pub mon_grouping: *mut ::c_char,
        pub positive_sign: *mut ::c_char,
        pub negative_sign: *mut ::c_char,
        pub int_frac_digits: ::c_char,
        pub frac_digits: ::c_char,
        pub p_cs_precedes: ::c_char,
        pub p_sep_by_space: ::c_char,
        pub n_cs_precedes: ::c_char,
        pub n_sep_by_space: ::c_char,
        pub p_sign_posn: ::c_char,
        pub n_sign_posn: ::c_char,
        pub int_p_cs_precedes: ::c_char,
        pub int_n_cs_precedes: ::c_char,
        pub int_p_sep_by_space: ::c_char,
        pub int_n_sep_by_space: ::c_char,
        pub int_p_sign_posn: ::c_char,
        pub int_n_sign_posn: ::c_char,
    }
}

pub const LC_COLLATE_MASK: ::c_int = (1 << 0);
pub const LC_CTYPE_MASK: ::c_int = (1 << 1);
pub const LC_MESSAGES_MASK: ::c_int = (1 << 2);
pub const LC_MONETARY_MASK: ::c_int = (1 << 3);
pub const LC_NUMERIC_MASK: ::c_int = (1 << 4);
pub const LC_TIME_MASK: ::c_int = (1 << 5);
pub const LC_ALL_MASK: ::c_int = LC_COLLATE_MASK
                               | LC_CTYPE_MASK
                               | LC_MESSAGES_MASK
                               | LC_MONETARY_MASK
                               | LC_NUMERIC_MASK
                               | LC_TIME_MASK;

pub const ERA: ::nl_item = 52;
pub const ERA_D_FMT: ::nl_item = 53;
pub const ERA_D_T_FMT: ::nl_item = 54;
pub const ERA_T_FMT: ::nl_item = 55;
pub const ALT_DIGITS: ::nl_item = 56;

pub const D_MD_ORDER: ::nl_item = 57;

pub const ALTMON_1: ::nl_item = 58;
pub const ALTMON_2: ::nl_item = 59;
pub const ALTMON_3: ::nl_item = 60;
pub const ALTMON_4: ::nl_item = 61;
pub const ALTMON_5: ::nl_item = 62;
pub const ALTMON_6: ::nl_item = 63;
pub const ALTMON_7: ::nl_item = 64;
pub const ALTMON_8: ::nl_item = 65;
pub const ALTMON_9: ::nl_item = 66;
pub const ALTMON_10: ::nl_item = 67;
pub const ALTMON_11: ::nl_item = 68;
pub const ALTMON_12: ::nl_item = 69;

pub const KERN_RND: ::c_int = 31;

extern {
    pub fn nl_langinfo_l(item: ::nl_item, locale: ::locale_t) -> *mut ::c_char;
    pub fn duplocale(base: ::locale_t) -> ::locale_t;
    pub fn freelocale(loc: ::locale_t) -> ::c_int;
    pub fn newlocale(mask: ::c_int,
                     locale: *const ::c_char,
                     base: ::locale_t) -> ::locale_t;
    pub fn uselocale(loc: ::locale_t) -> ::locale_t;
    pub fn querylocale(mask: ::c_int, loc: ::locale_t) -> *const ::c_char;
}
