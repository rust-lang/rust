#include"iostream"
using namespace std;
int gcd(int a,int b){
if(b==0)
    return a;
else
    return gcd(b,b%a);
}
int main(){
int a,b;
cout<<"Enter the value of a and b: "<<endl;
cin>>a>>b;
cout<<(int)gcd(a,b);
}
