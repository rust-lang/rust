#![crate_name = "foo"]

//@ has 'foo/struct.Class.html'
//@ has - '//*[@id="impl-ThreadSafe-for-T"]' 'impl<T> ThreadSafe for T'
pub struct Class<Span> {
    pub body: std::sync::Arc<[stmt::Statement<Span>]>,
    pub decorators: std::sync::Arc<[expr::Expression<Span>]>,
    pub span: Span,
}

pub trait ThreadSafe: Send + Sync {}

// With this line commented out, the slow behavior does not occur.
impl<T: Send + Sync> ThreadSafe for T {}

mod stmt {
    pub struct S00<Span>(pub Statement<Span>);
    pub struct S01<Span>(pub Statement<Span>);
    pub struct S02<Span>(pub Statement<Span>);
    pub struct S03<Span>(pub Statement<Span>);
    pub struct S04<Span>(pub Statement<Span>);
    pub struct S05<Span>(pub Statement<Span>);
    pub struct S06<Span>(pub Statement<Span>);
    pub struct S07<Span>(pub Statement<Span>);
    pub struct S08<Span>(pub Statement<Span>);
    pub struct S09<Span>(pub Statement<Span>);
    pub struct S10<Span>(pub Statement<Span>);
    pub struct S11<Span>(pub Statement<Span>);
    pub struct S12<Span>(pub Statement<Span>);
    pub struct S13<Span>(pub Statement<Span>);
    pub struct S14<Span>(pub Statement<Span>);
    pub struct S15<Span>(pub Statement<Span>);
    pub struct S16<Span>(pub Statement<Span>);
    pub struct S17<Span>(pub Statement<Span>);
    pub struct S18<Span>(pub Statement<Span>);
    pub struct S19<Span>(pub Statement<Span>);
    pub struct S20<Span>(pub Statement<Span>);
    pub struct S21<Span>(pub Statement<Span>);
    pub struct S22<Span>(pub Statement<Span>);
    pub struct S23<Span>(pub Statement<Span>);
    pub struct S24<Span>(pub Statement<Span>);
    pub struct S25<Span>(pub Statement<Span>);
    pub struct S26<Span>(pub Statement<Span>);
    pub struct S27<Span>(pub Statement<Span>);
    pub struct S28<Span>(pub Statement<Span>);
    pub struct S29<Span>(pub Statement<Span>);
    pub struct S30<Span>(pub Statement<Span>);
    pub struct S31<Span>(pub Statement<Span>);
    pub struct S32<Span>(pub Statement<Span>);
    pub struct S33<Span>(pub Statement<Span>);
    pub struct S34<Span>(pub Statement<Span>);
    pub struct S35<Span>(pub Statement<Span>);
    pub struct S36<Span>(pub Statement<Span>);
    pub struct S37<Span>(pub Statement<Span>);
    pub struct S38<Span>(pub Statement<Span>);
    pub struct S39<Span>(pub Statement<Span>);
    pub struct S40<Span>(pub Statement<Span>);
    pub struct S41<Span>(pub Statement<Span>);
    pub struct S42<Span>(pub Statement<Span>);
    pub struct S43<Span>(pub Statement<Span>);
    pub struct S44<Span>(pub Statement<Span>);
    pub struct S45<Span>(pub Statement<Span>);
    pub struct S46<Span>(pub Statement<Span>);
    pub struct S47<Span>(pub Statement<Span>);
    pub struct S48<Span>(pub Statement<Span>);
    pub struct S49<Span>(pub Statement<Span>);
    pub struct S50<Span>(pub Statement<Span>);
    pub struct S51<Span>(pub Statement<Span>);
    pub struct S52<Span>(pub Statement<Span>);
    pub struct S53<Span>(pub Statement<Span>);
    pub struct S54<Span>(pub Statement<Span>);
    pub struct S55<Span>(pub Statement<Span>);
    pub struct S56<Span>(pub Statement<Span>);
    pub struct S57<Span>(pub Statement<Span>);
    pub struct S58<Span>(pub Statement<Span>);
    pub struct S59<Span>(pub Statement<Span>);

    pub enum Statement<Span> {
        Class(std::sync::Arc<crate::Class<Span>>),
        S00(std::sync::Arc<S00<Span>>),
        S01(std::sync::Arc<S01<Span>>),
        S02(std::sync::Arc<S02<Span>>),
        S03(std::sync::Arc<S03<Span>>),
        S04(std::sync::Arc<S04<Span>>),
        S05(std::sync::Arc<S05<Span>>),
        S06(std::sync::Arc<S06<Span>>),
        S07(std::sync::Arc<S07<Span>>),
        S08(std::sync::Arc<S08<Span>>),
        S09(std::sync::Arc<S09<Span>>),
        S10(std::sync::Arc<S10<Span>>),
        S11(std::sync::Arc<S11<Span>>),
        S12(std::sync::Arc<S12<Span>>),
        S13(std::sync::Arc<S13<Span>>),
        S14(std::sync::Arc<S14<Span>>),
        S15(std::sync::Arc<S15<Span>>),
        S16(std::sync::Arc<S16<Span>>),
        S17(std::sync::Arc<S17<Span>>),
        S18(std::sync::Arc<S18<Span>>),
        S19(std::sync::Arc<S19<Span>>),
        S20(std::sync::Arc<S20<Span>>),
        S21(std::sync::Arc<S21<Span>>),
        S22(std::sync::Arc<S22<Span>>),
        S23(std::sync::Arc<S23<Span>>),
        S24(std::sync::Arc<S24<Span>>),
        S25(std::sync::Arc<S25<Span>>),
        S26(std::sync::Arc<S26<Span>>),
        S27(std::sync::Arc<S27<Span>>),
        S28(std::sync::Arc<S28<Span>>),
        S29(std::sync::Arc<S29<Span>>),
        S30(std::sync::Arc<S30<Span>>),
        S31(std::sync::Arc<S31<Span>>),
        S32(std::sync::Arc<S32<Span>>),
        S33(std::sync::Arc<S33<Span>>),
        S34(std::sync::Arc<S34<Span>>),
        S35(std::sync::Arc<S35<Span>>),
        S36(std::sync::Arc<S36<Span>>),
        S37(std::sync::Arc<S37<Span>>),
        S38(std::sync::Arc<S38<Span>>),
        S39(std::sync::Arc<S39<Span>>),
        S40(std::sync::Arc<S40<Span>>),
        S41(std::sync::Arc<S41<Span>>),
        S42(std::sync::Arc<S42<Span>>),
        S43(std::sync::Arc<S43<Span>>),
        S44(std::sync::Arc<S44<Span>>),
        S45(std::sync::Arc<S45<Span>>),
        S46(std::sync::Arc<S46<Span>>),
        S47(std::sync::Arc<S47<Span>>),
        S48(std::sync::Arc<S48<Span>>),
        S49(std::sync::Arc<S49<Span>>),
        S50(std::sync::Arc<S50<Span>>),
        S51(std::sync::Arc<S51<Span>>),
        S52(std::sync::Arc<S52<Span>>),
        S53(std::sync::Arc<S53<Span>>),
        S54(std::sync::Arc<S54<Span>>),
        S55(std::sync::Arc<S55<Span>>),
        S56(std::sync::Arc<S56<Span>>),
        S57(std::sync::Arc<S57<Span>>),
        S58(std::sync::Arc<S58<Span>>),
        S59(std::sync::Arc<S59<Span>>),
    }
}

mod expr {
    pub struct E00<Span>(pub Expression<Span>);
    pub struct E01<Span>(pub Expression<Span>);
    pub struct E02<Span>(pub Expression<Span>);
    pub struct E03<Span>(pub Expression<Span>);
    pub struct E04<Span>(pub Expression<Span>);
    pub struct E05<Span>(pub Expression<Span>);
    pub struct E06<Span>(pub Expression<Span>);
    pub struct E07<Span>(pub Expression<Span>);
    pub struct E08<Span>(pub Expression<Span>);
    pub struct E09<Span>(pub Expression<Span>);
    pub struct E10<Span>(pub Expression<Span>);
    pub struct E11<Span>(pub Expression<Span>);
    pub struct E12<Span>(pub Expression<Span>);
    pub struct E13<Span>(pub Expression<Span>);
    pub struct E14<Span>(pub Expression<Span>);
    pub struct E15<Span>(pub Expression<Span>);
    pub struct E16<Span>(pub Expression<Span>);
    pub struct E17<Span>(pub Expression<Span>);
    pub struct E18<Span>(pub Expression<Span>);
    pub struct E19<Span>(pub Expression<Span>);
    pub struct E20<Span>(pub Expression<Span>);
    pub struct E21<Span>(pub Expression<Span>);
    pub struct E22<Span>(pub Expression<Span>);
    pub struct E23<Span>(pub Expression<Span>);
    pub struct E24<Span>(pub Expression<Span>);
    pub struct E25<Span>(pub Expression<Span>);
    pub struct E26<Span>(pub Expression<Span>);
    pub struct E27<Span>(pub Expression<Span>);
    pub struct E28<Span>(pub Expression<Span>);
    pub struct E29<Span>(pub Expression<Span>);
    pub struct E30<Span>(pub Expression<Span>);
    pub struct E31<Span>(pub Expression<Span>);
    pub struct E32<Span>(pub Expression<Span>);
    pub struct E33<Span>(pub Expression<Span>);
    pub struct E34<Span>(pub Expression<Span>);
    pub struct E35<Span>(pub Expression<Span>);
    pub struct E36<Span>(pub Expression<Span>);
    pub struct E37<Span>(pub Expression<Span>);
    pub struct E38<Span>(pub Expression<Span>);
    pub struct E39<Span>(pub Expression<Span>);
    pub struct E40<Span>(pub Expression<Span>);
    pub struct E41<Span>(pub Expression<Span>);
    pub struct E42<Span>(pub Expression<Span>);
    pub struct E43<Span>(pub Expression<Span>);
    pub struct E44<Span>(pub Expression<Span>);
    pub struct E45<Span>(pub Expression<Span>);
    pub struct E46<Span>(pub Expression<Span>);
    pub struct E47<Span>(pub Expression<Span>);
    pub struct E48<Span>(pub Expression<Span>);
    pub struct E49<Span>(pub Expression<Span>);
    pub struct E50<Span>(pub Expression<Span>);
    pub struct E51<Span>(pub Expression<Span>);
    pub struct E52<Span>(pub Expression<Span>);
    pub struct E53<Span>(pub Expression<Span>);
    pub struct E54<Span>(pub Expression<Span>);
    pub struct E55<Span>(pub Expression<Span>);
    pub struct E56<Span>(pub Expression<Span>);
    pub struct E57<Span>(pub Expression<Span>);
    pub struct E58<Span>(pub Expression<Span>);
    pub struct E59<Span>(pub Expression<Span>);

    pub enum Expression<Span> {
        Class(std::sync::Arc<crate::Class<Span>>),
        E00(std::sync::Arc<E00<Span>>),
        E01(std::sync::Arc<E01<Span>>),
        E02(std::sync::Arc<E02<Span>>),
        E03(std::sync::Arc<E03<Span>>),
        E04(std::sync::Arc<E04<Span>>),
        E05(std::sync::Arc<E05<Span>>),
        E06(std::sync::Arc<E06<Span>>),
        E07(std::sync::Arc<E07<Span>>),
        E08(std::sync::Arc<E08<Span>>),
        E09(std::sync::Arc<E09<Span>>),
        E10(std::sync::Arc<E10<Span>>),
        E11(std::sync::Arc<E11<Span>>),
        E12(std::sync::Arc<E12<Span>>),
        E13(std::sync::Arc<E13<Span>>),
        E14(std::sync::Arc<E14<Span>>),
        E15(std::sync::Arc<E15<Span>>),
        E16(std::sync::Arc<E16<Span>>),
        E17(std::sync::Arc<E17<Span>>),
        E18(std::sync::Arc<E18<Span>>),
        E19(std::sync::Arc<E19<Span>>),
        E20(std::sync::Arc<E20<Span>>),
        E21(std::sync::Arc<E21<Span>>),
        E22(std::sync::Arc<E22<Span>>),
        E23(std::sync::Arc<E23<Span>>),
        E24(std::sync::Arc<E24<Span>>),
        E25(std::sync::Arc<E25<Span>>),
        E26(std::sync::Arc<E26<Span>>),
        E27(std::sync::Arc<E27<Span>>),
        E28(std::sync::Arc<E28<Span>>),
        E29(std::sync::Arc<E29<Span>>),
        E30(std::sync::Arc<E30<Span>>),
        E31(std::sync::Arc<E31<Span>>),
        E32(std::sync::Arc<E32<Span>>),
        E33(std::sync::Arc<E33<Span>>),
        E34(std::sync::Arc<E34<Span>>),
        E35(std::sync::Arc<E35<Span>>),
        E36(std::sync::Arc<E36<Span>>),
        E37(std::sync::Arc<E37<Span>>),
        E38(std::sync::Arc<E38<Span>>),
        E39(std::sync::Arc<E39<Span>>),
        E40(std::sync::Arc<E40<Span>>),
        E41(std::sync::Arc<E41<Span>>),
        E42(std::sync::Arc<E42<Span>>),
        E43(std::sync::Arc<E43<Span>>),
        E44(std::sync::Arc<E44<Span>>),
        E45(std::sync::Arc<E45<Span>>),
        E46(std::sync::Arc<E46<Span>>),
        E47(std::sync::Arc<E47<Span>>),
        E48(std::sync::Arc<E48<Span>>),
        E49(std::sync::Arc<E49<Span>>),
        E50(std::sync::Arc<E50<Span>>),
        E51(std::sync::Arc<E51<Span>>),
        E52(std::sync::Arc<E52<Span>>),
        E53(std::sync::Arc<E53<Span>>),
        E54(std::sync::Arc<E54<Span>>),
        E55(std::sync::Arc<E55<Span>>),
        E56(std::sync::Arc<E56<Span>>),
        E57(std::sync::Arc<E57<Span>>),
        E58(std::sync::Arc<E58<Span>>),
        E59(std::sync::Arc<E59<Span>>),
    }
}
