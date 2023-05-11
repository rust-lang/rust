// build-pass (FIXME(62277): could be check-pass?)
// pp-exact - Make sure we actually print the attributes
// pretty-expanded FIXME #23616

#![allow(non_camel_case_types)]
#![feature(rustc_attrs)]

enum crew_of_enterprise_d {

    #[rustc_dummy]
    jean_luc_picard,

    #[rustc_dummy]
    william_t_riker,

    #[rustc_dummy]
    beverly_crusher,

    #[rustc_dummy]
    deanna_troi,

    #[rustc_dummy]
    data,

    #[rustc_dummy]
    worf,

    #[rustc_dummy]
    geordi_la_forge,
}

fn boldly_go(_crew_member: crew_of_enterprise_d, _where: String) { }

fn main() {
    boldly_go(crew_of_enterprise_d::worf,
              "where no one has gone before".to_string());
}
