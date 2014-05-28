// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pp-exact - Make sure we actually print the attributes

enum crew_of_enterprise_d {

    #[captain]
    jean_luc_picard,

    #[oldcommander]
    william_t_riker,

    #[chief_medical_officer]
    beverly_crusher,

    #[ships_councellor]
    deanna_troi,

    #[lieutenant_oldcommander]
    data,

    #[chief_of_security]
    worf,

    #[chief_engineer]
    geordi_la_forge,
}

fn boldly_go(_crew_member: crew_of_enterprise_d, _where: String) { }

pub fn main() { boldly_go(worf, "where no one has gone before".to_string()); }
