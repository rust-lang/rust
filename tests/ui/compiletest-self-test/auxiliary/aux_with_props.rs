//@ edition: 2024
//@ compile-flags: --cfg=this_is_aux

pub fn aux_directives_are_respected() -> bool {
    cfg!(this_is_aux)
}
