// #2398
pub mod outer_mod {
    pub mod inner_mod {
       pub ( in outer_mod ) fn outer_mod_visible_fn() {}
         pub ( super ) fn super_mod_visible_fn() {}
      pub ( self ) fn inner_mod_visible_fn() {}
    }
}
