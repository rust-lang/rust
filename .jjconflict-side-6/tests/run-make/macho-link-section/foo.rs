#[unsafe(no_mangle)]
#[unsafe(link_section = "__TEXT,custom_code,regular,pure_instructions")]
static CODE: [u8; 10] = *b"0123456789";

#[unsafe(no_mangle)]
#[unsafe(link_section = "__DATA,all_attributes,regular,pure_instructions\
                        +no_toc+strip_static_syms+no_dead_strip+live_support\
                        +self_modifying_code+debug")]
static ALL_THE_ATTRIBUTES: u32 = 42;

#[unsafe(no_mangle)]
#[unsafe(link_section = "__DATA,__mod_init_func,mod_init_funcs")]
static CONSTRUCTOR: extern "C" fn() = constructor;
extern "C" fn constructor() {}
