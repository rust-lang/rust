; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

%Matrix = type { double*, i64, i64 }
%Base = type { i8 }

define void @caller(%Matrix* %W, %Matrix* %Wp, %Base* %M, %Base* %Mp) {
entry:
  %call11 = call fast double @__enzyme_autodiff(i8* bitcast (double (%Matrix*, %Base*)* @todiff to i8*), %Matrix* %W, %Matrix* %Wp, %Base* %M, %Base* %Mp)
  ret void
}

declare dso_local double @__enzyme_autodiff(i8*, %Matrix*, %Matrix*, %Base*, %Base*)

define linkonce_odr dso_local double @todiff(%Matrix* %dst, %Base* %lhs) {
entry:
  %call = tail call { %Matrix*, %Matrix* } @structret(%Base* %lhs)
  %res = call double @spoiler({ %Matrix*, %Matrix* } %call)
  ret double %res
}

define double @spoiler({ %Matrix*, %Matrix* }  %xpr) {
entry:
  ret double 0.000000e+00
}

define { %Matrix*, %Matrix* } @structret(%Base* %this) {
entry:
  %retval = alloca { %Matrix*, %Matrix* }, align 8
  %call = call %Matrix* @castBase13ToMatrix6(%Base* %this)
  %res = call { %Matrix*, %Matrix* } @noop({ %Matrix*, %Matrix* }* %retval)
  ret { %Matrix*, %Matrix* } %res
}

define %Matrix* @castBase13ToMatrix6(%Base* %this) {
entry:
  %0 = bitcast %Base* %this to %Matrix*
  ret %Matrix* %0
}

define { %Matrix*, %Matrix* } @noop({ %Matrix*, %Matrix* }* %this) {
entry:
  %res = load { %Matrix*, %Matrix* }, { %Matrix*, %Matrix* }* %this
  ret { %Matrix*, %Matrix* } %res
}

; CHECK: diffetodiff
