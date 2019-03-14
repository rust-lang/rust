// Ensure this code doesn't stack overflow.
// aux-build:enum-primitive.rs

#[macro_use] extern crate enum_primitive;

enum_from_primitive! {
    pub enum Test {
        A1,A2,A3,A4,A5,A6,
        B1,B2,B3,B4,B5,B6,
        C1,C2,C3,C4,C5,C6,
        D1,D2,D3,D4,D5,D6,
        E1,E2,E3,E4,E5,E6,
        F1,F2,F3,F4,F5,F6,
        G1,G2,G3,G4,G5,G6,
        H1,H2,H3,H4,H5,H6,
        I1,I2,I3,I4,I5,I6,
        J1,J2,J3,J4,J5,J6,
        K1,K2,K3,K4,K5,K6,
        L1,L2,L3,L4,L5,L6,
        M1,M2,M3,M4,M5,M6,
        N1,N2,N3,N4,N5,N6,
        O1,O2,O3,O4,O5,O6,
        P1,P2,P3,P4,P5,P6,
        Q1,Q2,Q3,Q4,Q5,Q6,
        R1,R2,R3,R4,R5,R6,
        S1,S2,S3,S4,S5,S6,
        T1,T2,T3,T4,T5,T6,
        U1,U2,U3,U4,U5,U6,
        V1,V2,V3,V4,V5,V6,
        W1,W2,W3,W4,W5,W6,
        X1,X2,X3,X4,X5,X6,
        Y1,Y2,Y3,Y4,Y5,Y6,
        Z1,Z2,Z3,Z4,Z5,Z6,
    }
}
