set -x

compiler=`echo $1|sed -e 's/\//\\\\\//g'`

sed "s/compiler_name = .*/compiler_name = \"$compiler\";/g" -i c_check
sed "s/flags = .*/flags = \"\";/g" -i c_check
sed "s/all: getarch_2nd/all: \$(TARGET_CONF) dummy/g" -i Makefile.prebuild
sed "s/f_check getarch/f_check/g" -i Makefile.prebuild
sed "/getarch/d" -i Makefile.prebuild
sed "/avx512/d" -i Makefile.prebuild
sed "s/# GEMM_MULTI/GEMM_MULTI/g" -i Makefile.rule
sed "s/COMMON_OPT = -O2/COMMON_OPT =/g" -i Makefile.system
sed "/#define GEMM_P/d" -i common_param.h
sed "/#define GEMM_Q/d" -i common_param.h
echo > exports/gensymbol
