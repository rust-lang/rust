macro_rules! imp {
    ($name:ident) => {
        pub struct $name {
            pub op: usize,
        }
        impl $name {
            pub fn op() {}
            pub fn cmp() {}
            pub fn map() {}
            pub fn pop() {}
            pub fn ptr() {}
            pub fn rpo() {}
            pub fn drop() {}
            pub fn copy() {}
            pub fn zip() {}
            pub fn sup() {}
            pub fn pa() {}
            pub fn pb() {}
            pub fn pc() {}
            pub fn pd() {}
            pub fn pe() {}
            pub fn pf() {}
            pub fn pg() {}
            pub fn ph() {}
            pub fn pi() {}
            pub fn pj() {}
            pub fn pk() {}
            pub fn pl() {}
            pub fn pm() {}
            pub fn pn() {}
            pub fn po() {}
        }
    };
    ($name:ident, $($names:ident),*) => {
        imp!($name);
        imp!($($names),*);
    };
}
macro_rules! en {
    ($name:ident) => {
        pub enum $name {
            Ptr,
            Rp,
            Rpo,
            Pt,
            Drop,
            Dr,
            Dro,
            Sup,
            Op,
            Cmp,
            Map,
            Mp,
        }
    };
    ($name:ident, $($names:ident),*) => {
        en!($name);
        en!($($names),*);
    };
}

imp!(Ot, Foo, Cmp, Map, Loc, Lac, Toc, Si, Sig, Sip, Psy, Psi, Py, Pi, Pa, Pb, Pc, Pd);
imp!(Pe, Pf, Pg, Ph, Pj, Pk, Pl, Pm, Pn, Po, Pq, Pr, Ps, Pt, Pu, Pv, Pw, Px, Pz, Ap, Bp, Cp);
imp!(Dp, Ep, Fp, Gp, Hp, Ip, Jp, Kp, Lp, Mp, Np, Op, Pp, Qp, Rp, Sp, Tp, Up, Vp, Wp, Xp, Yp, Zp);

en!(Place, Plac, Plae, Plce, Pace, Scalar, Scalr, Scaar, Sclar, Salar);

pub struct P;

pub struct VeryLongTypeName;
impl VeryLongTypeName {
    pub fn p() {}
    pub fn ap() {}
}
