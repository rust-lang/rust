// rustfmt-where_layout: Horizontal
// rustfmt-error_on_line_overflow: false
// Where layout

fn lorem<Ipsum, Dolor>(ipsum: Ipsum, dolor: Dolor)
    where Ipsum: IpsumDolorSitAmet, Dolor: DolorSitAmetConsectetur
{
    // body
}

fn lorem<Ipsum, Dolor, Sit, Amet>(ipsum: Ipsum, dolor: Dolor, sit: Sit, amet: Amet)
    where Ipsum: IpsumDolorSitAmet, Dolor: DolorSitAmetConsectetur, Sit: SitAmetConsecteturAdipiscing, Amet: AmetConsecteturAdipiscingElit
{
    // body
}
