#![allow(unused_attributes)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
// pp-exact - Make sure we actually print the attributes
// pretty-expanded FIXME #23616

#![feature(custom_attribute)]

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

pub fn main() {
    boldly_go(crew_of_enterprise_d::worf,
              "where no one has gone before".to_string());
}
