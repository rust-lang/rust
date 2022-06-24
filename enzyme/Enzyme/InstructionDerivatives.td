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
