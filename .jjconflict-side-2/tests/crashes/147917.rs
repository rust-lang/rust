//@ known-bug: #147917
#![feature(custom_inner_attributes)]
impl[u8;
    #[cfg_attr]({
        #[cfg_attr] {
            {
                #!
            }
        } !
    })
] {
    #![cfg_eval]
}
