// pp-exact - Make sure we actually print the attributes

enum crew_of_enterprise_d {

    #[captain]
    jean_luc_picard,

    #[commander]
    william_t_riker,

    #[chief_medical_officer]
    beverly_crusher,

    #[ships_councellor]
    deanna_troi,

    #[lieutenant_commander]
    data,

    #[chief_of_security]
    worf,

    #[chief_engineer]
    geordi_la_forge,
}

fn boldly_go(_crew_member: crew_of_enterprise_d, _where: str) { }

fn main() { boldly_go(worf, "where no one has gone before"); }
