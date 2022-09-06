class Type;
class CallPattern<dag patternToMatch, list<string> funcNames, list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<string> names = funcNames;
  list<dag> ArgDerivatives = resultOps;
}

class Inst<string mnemonic> {
  string name = mnemonic;
}

// Define ops to rewrite.
def T1: Type;

def FDiv : Inst<"FDiv">;
def FAdd : Inst<"FAdd">;
def FSub : Inst<"FSub">;
def FMul : Inst<"FMul">;
def FNeg : Inst<"FNeg">;

def DifR : Inst<"DifR">;

// Same function as the one being called
def SameFunc {
}

// Same types / attributes / callingconv as original
// Different name, however
class SameTypesFunc<string name_> {
  string name = name_;
}

class Attribute<string name_> {
  string name = name_;
}

// So we can also extend it by not input related 
// types, eg. for mem related Intrinsics
class TypeInfo {
}

class TypeOf<string val> : TypeInfo {
}

class Intrinsic<string name_, list<dag> types_> {
  string name = name_;
  list<dag> types = types_;
}

def ReadNone : Attribute<"ReadNone">;
def ReadOnly : Attribute<"ReadOnly">;
def NoUnwind : Attribute<"NoUnwind">;

class Call<dag mnemonic, list<Attribute> _fnattrs=[]> {
  dag func = mnemonic;
  list<Attribute> fnattrs = _fnattrs;
}

def Op {
}

class ConstantFP<string val> {
  string value = val;
}

class DiffeRet<string val> {
}

class Shadow<string val> {
}

def : CallPattern<(Op $x),
                  ["atan", "atanf", "atanl", "__fd_atan_1"],
                  [(FDiv (DiffeRet<"">), (FAdd (ConstantFP<"1.0"> $x), (FMul $x, $x)))]
                  >;

def : CallPattern<(Op $y, $x),
                  ["atan2"],
                  [
                  (FDiv (FMul (DiffeRet<"">), $x), (FAdd (FMul $x, $x), (FMul $y, $y))),
                  (FNeg (FDiv (FMul (DiffeRet<"">), $y), (FAdd (FMul $x, $x), (FMul $y, $y))))
                  ]
                  >;
def : CallPattern<(Op $x),
                  ["cbrt"],
                  [(FDiv (FMul (DiffeRet<"">), (Call<(SameFunc), [ReadNone,NoUnwind]> $x) ), (FMul (ConstantFP<"3.0"> $x), $x))]
                  >;

def : CallPattern<(Op $x, $y),
                  ["hypot", "hypotf", "hypotl"],
                  [
                    (FDiv (FMul (DiffeRet<"">), $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x, $y)),
                    (FDiv (FMul (DiffeRet<"">), $y), (Call<(SameFunc), [ReadNone,NoUnwind]> $x, $y))
                  ]
                  >;

def : CallPattern<(Op $x),
                  ["tanh"],
                  [(FDiv (DiffeRet<"">), (FMul(Call<(SameTypesFunc<"cosh">), [ReadNone,NoUnwind]> $x):$c, $c))]>;

def : CallPattern<(Op $x),
                  ["tanhf"],
                  [(FDiv (DiffeRet<"">), (FMul(Call<(SameTypesFunc<"coshf">), [ReadNone,NoUnwind]> $x):$c, $c))]>;

def : CallPattern<(Op $x),
                  ["cosh"],
                  [(FMul (DiffeRet<"">), (Call<(SameTypesFunc<"sinh">), [ReadNone,NoUnwind]> $x))]>;
def : CallPattern<(Op $x),
                  ["coshf"],
                  [(FMul (DiffeRet<"">), (Call<(SameTypesFunc<"sinhf">), [ReadNone,NoUnwind]> $x))]>;

def : CallPattern<(Op $x),
                  ["sinh"],
                  [(FMul (DiffeRet<"">), (Call<(SameTypesFunc<"cosh">), [ReadNone,NoUnwind]> $x))]>;
def : CallPattern<(Op $x),
                  ["sinhf"],
                  [(FMul (DiffeRet<"">), (Call<(SameTypesFunc<"coshf">), [ReadNone,NoUnwind]> $x))]>;

def : CallPattern<(Op $x),
                  ["exp10"],
                  [(FMul (FMul (DiffeRet<"">), (Call<(SameFunc), [ReadNone,NoUnwind]> $x) ), (ConstantFP<"2.30258509299404568401799145468"> $x))]
                  >;
def : CallPattern<(Op $x),
                  ["tan", "tanf", "tanl"],
                  [(FMul (DiffeRet<"">), (FAdd (ConstantFP<"1.0"> $x), (FMul(Call<(SameFunc), [ReadNone,NoUnwind]> $x):$c, $c)))]>;
def : CallPattern<(Op $x, $y),
                  ["remainder"],
                  [
                    (DiffeRet<"">),
                    (FMul (FNeg (DiffeRet<"">)), (Intrinsic<"round", [(TypeOf<""> $x)]> (FDiv $x, $y)))
                  ]
                  >;
def : CallPattern<(Op $x),
                  ["asin", "asinf", "asinl"],
                  [(FDiv (DiffeRet<"">), (Intrinsic<"sqrt", [(TypeOf<""> $x)]> (FSub (ConstantFP<"1.0"> $x), (FMul $x, $x)))  )]                  
                  >;
def : CallPattern<(Op $x),
                  ["log1p", "log1pf", "log1pl"],
                  [(FDiv (DiffeRet<"">), (FAdd (ConstantFP<"1.0"> $x), $x))]
                  >;
def : CallPattern<(Op $x),
                  ["expm1"],
                  [(FMul (Intrinsic<"exp", [(TypeOf<""> $x)]> $x), (DiffeRet<"">))]
                  >;

// Unnormalized sinc(x) = sin(x)/x
def : CallPattern<(Op $x),
                  ["sinc", "sincf", "sincl"],
                  [(FMul (DiffeRet<"">), (FDiv (FSub (Intrinsic<"cos", [(TypeOf<""> $x)]> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), $x))]>;

// Normalized sinc(x) = sin(pi x)/(pi x)
def : CallPattern<(Op $x),
                  ["sincn", "sincnf", "sincnl"],
                  [(FMul (DiffeRet<"">), (FDiv (FSub (Intrinsic<"cos", [(TypeOf<""> $x)]> (FMul (ConstantFP<"3.1415926535897962684626433"> $x), $x)), (Call<(SameFunc), [ReadNone,NoUnwind]> $x)), $x))]>;
