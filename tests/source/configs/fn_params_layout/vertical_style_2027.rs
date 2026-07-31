// rustfmt-style_edition: 2027
// rustfmt-fn_params_layout: Vertical
// Function arguments density

trait Lorem {
    fn lorem(ipsum: Ipsum);

    fn lorem(ipsum: Ipsum) -> Dolor;

    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet);

    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet) {
        // body
    }

    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet, consectetur: onsectetur, adipiscing: Adipiscing, elit: Elit);

    fn lorem(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet, consectetur: onsectetur, adipiscing: Adipiscing, elit: Elit) {
        // body
    }

    fn long_param_name(lorem_ipsum_dolor_sit_amet_consectetur_adipiscing_elit_sed_do_eiusmod: Tempor);

    fn long_param_type(lorem: IpsumDolorSitAmetConsecteturAdipiscingElitSedDoEiusmodTemporIncididuntUtLabore,
    );

    fn long_return_type(lorem: Lorem)
        -> IpsumDolorSitAmetConsecteturAdipiscingElitSedDoEiusmodTemporIncididuntUtLabore;

    fn lorem_ipsum_dolor_sit_amet_consectetur_adipiscing_elit_sed_do_eiusmod_tempor_incididunt
        (lorem: Lorem);

    fn lorem<T: IpsumDolorSitAmetConsecteturAdipiscingElitSedDoEiusmodTemporIncididuntUtLaboreEt>(t: T);

    fn lorem<T: IpsumDolorSitAmetConsecteturAdipiscingElitSedDoEiusmodTemporIncididuntUtLaboreEtDolore>
    (
        t: T,
    );
}
